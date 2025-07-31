#include "gann_hnsw.hpp"

GannHNSW::GannHNSW(
    size_t dimension,
    size_t max_elements,
    size_t M,
    size_t ef_construction,
    double ml)
    : dimension_(dimension),
      max_elements_(max_elements),
      M_(M),
      dmin_(M),     // dmin = M for NSW construction
      dmax_(M * 2), // dmax = 2*M to allow for bidirectional connections
      ef_construction_(ef_construction),
      ml_(ml),
      num_elements_(0),
      max_level_(16),
      level_generator_(std::random_device{}()),
      level_distribution_(0.0, 1.0)
{
    data_.reserve(max_elements);
    entry_point_ = std::numeric_limits<VertexId>::max();
}

/**
 * @brief Compute L2 distance between two vectors using AVX512 optimization
 * @param a First vector
 * @param b Second vector
 * @return L2 distance between vectors
 */
GannHNSW::Distance GannHNSW::computeDistance(const std::vector<float> &a, const std::vector<float> &b) const
{
    if (a.size() != b.size() || a.size() != dimension_)
    {
        throw std::invalid_argument("Vector dimensions must match and equal to index dimension");
    }

    const float *ptr_a = a.data();
    const float *ptr_b = b.data();
    const size_t dim = dimension_;

// Use AVX2 if available and dimension is large enough
#ifdef __AVX2__
    if (dim >= 8)
    {
        __m256 sum_vec = _mm256_setzero_ps();
        size_t avx2_ops = dim / 8;

        for (size_t i = 0; i < avx2_ops; ++i)
        {
            __m256 a_vec = _mm256_loadu_ps(ptr_a + i * 8);
            __m256 b_vec = _mm256_loadu_ps(ptr_b + i * 8);
            __m256 diff = _mm256_sub_ps(a_vec, b_vec);
            sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
        }

        // Horizontal sum
        alignas(32) float temp[8];
        _mm256_store_ps(temp, sum_vec);
        Distance result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

        // Handle remaining elements
        for (size_t i = avx2_ops * 8; i < dim; ++i)
        {
            Distance diff = ptr_a[i] - ptr_b[i];
            result += diff * diff;
        }

        return std::sqrt(result);
    }
#endif

// Use AVX512 if available and dimension is large enough
#ifdef __AVX512F__
    if (dim >= 16)
    {
        __m512 sum_vec = _mm512_setzero_ps();
        size_t avx512_ops = dim / 16;

        for (size_t i = 0; i < avx512_ops; ++i)
        {
            __m512 a_vec = _mm512_loadu_ps(ptr_a + i * 16);
            __m512 b_vec = _mm512_loadu_ps(ptr_b + i * 16);
            __m512 diff = _mm512_sub_ps(a_vec, b_vec);
            sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
        }

        // Horizontal sum of the 16 floats in sum_vec
        Distance result = _mm512_reduce_add_ps(sum_vec);

        // Handle remaining elements
        size_t remaining = dim % 16;
        for (size_t i = avx512_ops * 16; i < dim; ++i)
        {
            Distance diff = ptr_a[i] - ptr_b[i];
            result += diff * diff;
        }

        return std::sqrt(result);
    }
#endif

    // Fallback to scalar computation
    Distance sum = 0.0f;
    for (size_t i = 0; i < dim; ++i)
    {
        Distance diff = ptr_a[i] - ptr_b[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}

/**
 * @brief Generate random level for a new vertex following HNSW paper
 * @return Level number (0 = bottom layer, higher numbers = upper layers)
 */
size_t GannHNSW::getRandomLevel() const
{
    // Thread-local generator to avoid contention
    thread_local std::mt19937 local_generator(std::random_device{}());
    thread_local std::uniform_real_distribution<double> local_distribution(0.0, 1.0);

    double random_val = local_distribution(local_generator);
    // Avoid log(0) which would give -inf
    random_val = std::max(random_val, std::numeric_limits<double>::min());

    size_t level = static_cast<size_t>(-std::log(random_val) * ml_);
    return std::min(level, max_level_);
}

/**
 * @brief Build a single layer using GANN's divide-and-conquer strategy
 * @param level Layer level to build
 * @param vertices List of vertex IDs to include in this layer
 * @param num_threads Number of threads for parallel construction
 */
void GannHNSW::buildLayer(size_t level, const std::vector<VertexId> &vertices, size_t num_threads)
{
    if (vertices.empty())
        return;

    // Ensure we have this layer
    if (level >= layers_.size())
    {
        layers_.resize(level + 1);
    }

    Layer &layer = layers_[level];
    layer.level = level;
    layer.vertices.clear();
    layer.vertices.reserve(vertices.size());

    // Initialize vertices for this layer
    for (VertexId vid : vertices)
    {
        Vertex vertex;
        vertex.id = vid;
        vertex.neighbors.reserve(dmax_); // Reserve space for bidirectional links
        vertex.explored = false;
        vertex.distance_to_query = 0.0f;
        layer.vertices.push_back(vertex);
    }

    // Phase 1: Build local graphs in parallel
    buildLocalGraphsParallel(vertices, level, num_threads);

    // Phase 2: Merge local graphs (sequential for now, as per Algorithm 2)
    mergeLocalGraphs(level);
}

/**
 * @brief Build local graphs in parallel following GANN Algorithm 2. If num_threads is 0, use max number of threads available. This is Phase 1 of GANN's Algorithm 2.
 * @param vertices Vertex IDs to process
 * @param level Current layer level
 * @param num_threads Number of threads to use
 */
void GannHNSW::buildLocalGraphsParallel(const std::vector<VertexId> &vertices, size_t level, size_t num_threads)
{
    if (num_threads == 0)
    {
        num_threads = std::thread::hardware_concurrency();
    }

    // Partition vertices into groups for parallel processing
    const size_t vertices_per_thread = std::max(size_t(1), vertices.size() / num_threads);
    std::vector<std::thread> threads;

    // Use mutex to protect shared neighbor updates
    std::mutex neighbor_update_mutex;

    auto buildLocalGraph = [this, level, &neighbor_update_mutex](
                               const std::vector<VertexId> &local_vertices,
                               size_t start_idx,
                               size_t end_idx)
    {
        Layer &layer = layers_[level];

        for (size_t i = start_idx; i < end_idx && i < local_vertices.size(); ++i)
        {
            VertexId current_vid = local_vertices[i];

            // Find current vertex in layer
            auto vertex_it = std::find_if(layer.vertices.begin(), layer.vertices.end(),
                                          [current_vid](const Vertex &v)
                                          { return v.id == current_vid; });

            if (vertex_it == layer.vertices.end())
                continue;

            std::vector<std::pair<Distance, VertexId>> candidates;
            candidates.reserve(i);

            // Search among previously inserted vertices in this local group
            for (size_t j = start_idx; j < i; ++j)
            {
                VertexId neighbor_vid = local_vertices[j];
                Distance dist = computeDistance(data_[current_vid], data_[neighbor_vid]);
                candidates.emplace_back(dist, neighbor_vid);
            }

            std::sort(candidates.begin(), candidates.end());

            // Add forward edges using dmin_ (following GANN paper)
            size_t max_connections = std::min(dmin_, candidates.size());

            // Store backward edges to add later (thread-safe)
            std::vector<std::pair<VertexId, VertexId>> backward_edges;

            for (size_t k = 0; k < max_connections; ++k)
            {
                VertexId neighbor_id = candidates[k].second;
                vertex_it->neighbors.push_back(neighbor_id);
                backward_edges.emplace_back(neighbor_id, current_vid);
            }

            // Add backward edges with proper synchronization
            {
                std::lock_guard<std::mutex> lock(neighbor_update_mutex);
                for (const auto &[neighbor_id, current_id] : backward_edges)
                {
                    auto neighbor_it = std::find_if(layer.vertices.begin(), layer.vertices.end(),
                                                    [neighbor_id](const Vertex &v)
                                                    { return v.id == neighbor_id; });

                    // Skip if neighbor not found
                    if (neighbor_it == layer.vertices.end())
                        continue;

                    neighbor_it->neighbors.push_back(current_id);

                    // Pruning: keep only best dmax_ connections
                    if (neighbor_it->neighbors.size() <= dmax_)
                    {
                        continue;
                    }

                    // Recompute distances and keep best connections
                    std::vector<std::pair<Distance, VertexId>> neighbor_candidates;
                    for (VertexId nid : neighbor_it->neighbors)
                    {
                        Distance dist = computeDistance(data_[neighbor_id], data_[nid]);
                        neighbor_candidates.emplace_back(dist, nid);
                    }

                    std::sort(neighbor_candidates.begin(), neighbor_candidates.end());
                    neighbor_it->neighbors.clear();
                    neighbor_it->neighbors.reserve(dmax_);

                    for (size_t idx = 0; idx < std::min(dmax_, neighbor_candidates.size()); ++idx)
                    {
                        neighbor_it->neighbors.push_back(neighbor_candidates[idx].second);
                    }
                }
            }
        }
    };

    // Launch threads
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t start_idx = t * vertices_per_thread;
        size_t end_idx = (t == num_threads - 1) ? vertices.size() : (t + 1) * vertices_per_thread;

        if (start_idx >= vertices.size())
            continue;

        threads.emplace_back(buildLocalGraph, std::cref(vertices), start_idx, end_idx);
    }

    for (auto &thread : threads)
    {
        thread.join();
    }
}

/**
 * @brief Merge local graphs following GANN Algorithm 2 Phase 2
 * @param level Current layer level being processed
 */
void GannHNSW::mergeLocalGraphs(size_t level)
{
    if (level >= layers_.size())
        return;

    Layer &layer = layers_[level];
    if (layer.vertices.empty())
        return;

    // For now, implement a simplified merge strategy
    // of multiple local graphs built in parallel

    // Edge list for batch processing backward edges
    std::vector<std::pair<VertexId, VertexId>> edge_list;
    edge_list.reserve(layer.vertices.size() * M_);

    // Collect all edges that need backward updates
    for (const auto &vertex : layer.vertices)
    {
        for (VertexId neighbor_id : vertex.neighbors)
        {
            edge_list.emplace_back(neighbor_id, vertex.id);
        }
    }

    // Sort edges by starting vertex ID for efficient batch processing
    std::sort(edge_list.begin(), edge_list.end());

    // Process backward edges in batches (CSR format as mentioned in paper)
    for (size_t i = 0; i < edge_list.size();)
    {
        VertexId current_vertex = edge_list[i].first;

        // Find the vertex in our layer
        auto vertex_it = std::find_if(layer.vertices.begin(), layer.vertices.end(),
                                      [current_vertex](const Vertex &v)
                                      { return v.id == current_vertex; });

        if (vertex_it == layer.vertices.end())
        {
            ++i;
            continue;
        }

        // Collect all backward edges for this vertex
        std::vector<std::pair<Distance, VertexId>> candidates;
        size_t j = i;
        while (j < edge_list.size() && edge_list[j].first == current_vertex)
        {
            VertexId neighbor_id = edge_list[j].second;
            Distance dist = computeDistance(data_[current_vertex], data_[neighbor_id]);
            candidates.emplace_back(dist, neighbor_id);
            ++j;
        }

        // Sort by distance and keep only the best dmax_ connections
        std::sort(candidates.begin(), candidates.end());

        // Update the vertex's neighbor list with merged results
        vertex_it->neighbors.clear();
        vertex_it->neighbors.reserve(dmax_);

        size_t max_connections = std::min(static_cast<size_t>(dmax_), candidates.size());
        for (size_t k = 0; k < max_connections; ++k)
        {
            vertex_it->neighbors.push_back(candidates[k].second);
        }

        i = j; // Move to next vertex
    }
}

/**
 * @brief Search for nearest neighbors in a specific layer using GANN's lazy update strategy.
 * @param query Query vector
 * @param k Number of nearest neighbors to find
 * @param ef Search parameter (exploration factor)
 * @param level Layer level to search
 * @param entry_point Starting vertex ID for search
 * @return SearchResult containing found neighbors
 */
GannHNSW::SearchResult GannHNSW::searchLayer(
    const std::vector<float> &query,
    size_t k,
    size_t ef,
    size_t level,
    VertexId entry_point) const
{
    SearchResult result;
    if (level >= layers_.size() || entry_point >= num_elements_)
    {
        return result;
    }

    const Layer &layer = layers_[level];
    if (layer.vertices.empty())
    {
        return result;
    }

    // GANN search context with lazy update strategy
    SearchContext context;
    context.ln = std::max(k, ef); // Size of N array
    context.lt = dmax_;           // Size of T array (max neighbors)

    context.N.clear();
    context.T.clear();
    context.N.reserve(context.ln);
    context.T.reserve(context.lt);

    // Initialize with entry point
    Vertex entry_vertex;
    entry_vertex.id = entry_point;
    entry_vertex.distance_to_query = computeDistance(query, data_[entry_point]);
    entry_vertex.explored = false;
    context.N.push_back(entry_vertex);

    // GANN search iterations following Algorithm in paper
    while (true)
    {
        // Phase 1: Candidate locating - find first unexplored vertex
        auto unexplored_it = std::find_if(context.N.begin(), context.N.end(),
                                          [](const Vertex &v)
                                          { return !v.explored; });

        if (unexplored_it == context.N.end())
        {
            break; // All vertices explored, terminate
        }

        VertexId exploring_vertex = unexplored_it->id;
        unexplored_it->explored = true;

        // Phase 2: Explore neighbors
        context.T.clear();

        // Find the exploring vertex in the layer
        auto layer_vertex_it = std::find_if(layer.vertices.begin(), layer.vertices.end(),
                                            [exploring_vertex](const Vertex &v)
                                            { return v.id == exploring_vertex; });

        if (layer_vertex_it != layer.vertices.end())
        {
            // Load neighbors into T
            for (VertexId neighbor_id : layer_vertex_it->neighbors)
            {
                // Skip invalid neighbor IDs
                if (neighbor_id >= data_.size())
                {
                    continue;
                }

                Vertex neighbor;
                neighbor.id = neighbor_id;
                neighbor.explored = false;
                context.T.push_back(neighbor);
            }
        }

        // Phase 3: Bulk distance computation
        computeDistanceParallel(query, context.T);

        // Phase 4: Lazy check - mark vertices already in N as explored
        for (auto &t_vertex : context.T)
        {
            auto found_in_n = std::find_if(context.N.begin(), context.N.end(),
                                           [&t_vertex](const Vertex &n_vertex)
                                           { return n_vertex.id == t_vertex.id; });

            if (found_in_n != context.N.end())
            {
                t_vertex.explored = true;
            }
        }

        // Phase 5: Sorting T by distance
        std::sort(context.T.begin(), context.T.end(),
                  [](const Vertex &a, const Vertex &b)
                  {
                      return a.distance_to_query < b.distance_to_query;
                  });

        // Phase 6: Candidate update - merge T into N
        std::vector<Vertex> merged_candidates;
        merged_candidates.reserve(context.N.size() + context.T.size());

        // Merge N and T
        std::merge(context.N.begin(), context.N.end(),
                   context.T.begin(), context.T.end(),
                   std::back_inserter(merged_candidates),
                   [](const Vertex &a, const Vertex &b)
                   {
                       return a.distance_to_query < b.distance_to_query;
                   });

        // Keep only the best ln candidates
        if (merged_candidates.size() > context.ln)
        {
            merged_candidates.resize(context.ln);
        }

        context.N = std::move(merged_candidates);
    }

    // Extract results - return top k
    result.ids.reserve(k);
    result.distances.reserve(k);

    size_t return_count = std::min(k, context.N.size());
    for (size_t i = 0; i < return_count; ++i)
    {
        result.ids.push_back(context.N[i].id);
        result.distances.push_back(context.N[i].distance_to_query);
    }

    return result;
}

/**
 * @brief Compute distances in parallel for a batch of candidates (GANN Algorithm Phase 3)
 * @param query Query vector
 * @param candidates Vector of candidates to compute distances for
 */
void GannHNSW::computeDistanceParallel(const std::vector<float> &query, std::vector<Vertex> &candidates) const
{
    if (candidates.empty())
        return;

    // For CPU implementation, we can use simple parallel for loop
    // In GPU version, this would be massively parallel
    const size_t num_candidates = candidates.size();
    const size_t num_threads = std::thread::hardware_concurrency();

    if (num_candidates < num_threads * 4)
    {
        // Too few candidates for threading overhead
        for (auto &candidate : candidates)
        {
            candidate.distance_to_query = computeDistance(query, data_[candidate.id]);
        }
        return;
    }

    std::vector<std::thread> threads;
    const size_t candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t start_idx = t * candidates_per_thread;
        size_t end_idx = std::min(start_idx + candidates_per_thread, num_candidates);

        if (start_idx >= num_candidates)
            break;

        threads.emplace_back([this, &query, &candidates, start_idx, end_idx]()
                             {
            for (size_t i = start_idx; i < end_idx; ++i) {
                candidates[i].distance_to_query = computeDistance(query, data_[candidates[i].id]);
            } });
    }

    for (auto &thread : threads)
    {
        thread.join();
    }
}

/**
 * @brief Wrapper function to build HNSW index using GANN's divide-and-conquer approach
 * @param input_data 2D vector of input data points
 * @param num_threads Number of threads for parallel construction
 */
void GannHNSW::build(const std::vector<std::vector<float>> &input_data, size_t num_threads)
{
    if (input_data.empty())
    {
        throw std::invalid_argument("Input data cannot be empty");
    }

    if (input_data[0].size() != dimension_)
    {
        throw std::invalid_argument("Input data dimension mismatch");
    }

    if (num_threads == 0)
    {
        num_threads = std::thread::hardware_concurrency();
    }

    // Store data
    data_ = input_data;
    num_elements_ = data_.size();

    // Generate levels for all vertices following HNSW paper
    std::vector<std::vector<VertexId>> level_vertices(max_level_ + 1);
    VertexId highest_level_vertex = 0;
    size_t highest_level = 0;

    for (VertexId vid = 0; vid < num_elements_; ++vid)
    {
        size_t vertex_level = getRandomLevel();

        // Track highest level vertex for entry point
        if (vertex_level > highest_level)
        {
            highest_level = vertex_level;
            highest_level_vertex = vid;
        }

        // Add vertex to all levels from 0 to vertex_level
        for (size_t level = 0; level <= vertex_level; ++level)
        {
            level_vertices[level].push_back(vid);
        }
    }

    // Set entry point to highest level vertex
    entry_point_ = highest_level_vertex;

    // Build each layer using GANN's Algorithm 2
    for (size_t level = 0; level < level_vertices.size(); ++level)
    {
        if (!level_vertices[level].empty())
        {
            buildLayer(level, level_vertices[level], num_threads);
        }
    }
}

/**
 * @brief Wrapper function to search for k nearest neighbors using GANN's lazy update strategy across multiple queries
 * @param query_data 2D vector of query vectors
 * @param k Number of nearest neighbors to return
 * @param ef Search parameter (size of dynamic candidate list)
 * @param num_threads Number of threads for parallel search
 * @return SearchResult containing neighbor IDs and distances for all queries
 */
GannHNSW::SearchResult GannHNSW::search(const std::vector<std::vector<float>> &query_data, size_t k, size_t ef, size_t num_threads) const
{
    SearchResult final_result;

    if (query_data.empty() || num_elements_ == 0)
    {
        return final_result;
    }

    if (num_threads == 0)
    {
        num_threads = std::thread::hardware_concurrency();
    }

    const size_t num_queries = query_data.size();

    // Prepare result containers
    std::vector<std::vector<VertexId>> all_ids(num_queries);
    std::vector<std::vector<Distance>> all_distances(num_queries);

    // Process queries in parallel
    if (num_queries >= num_threads)
    {
        // Parallel processing for multiple queries
        std::vector<std::thread> threads;
        const size_t queries_per_thread = (num_queries + num_threads - 1) / num_threads;

        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start_idx = t * queries_per_thread;
            size_t end_idx = std::min(start_idx + queries_per_thread, num_queries);

            if (start_idx >= num_queries)
                break;

            threads.emplace_back([this, &query_data, &all_ids, &all_distances, start_idx, end_idx, k, ef]()
                                 {
                for (size_t q = start_idx; q < end_idx; ++q) {
                    SearchResult single_result = searchQuery(query_data[q], k, ef);
                    all_ids[q] = std::move(single_result.ids);
                    all_distances[q] = std::move(single_result.distances);
                } });
        }

        for (auto &thread : threads)
        {
            thread.join();
        }
    }
    else
    {
        // Sequential processing for few queries
        for (size_t q = 0; q < num_queries; ++q)
        {
            SearchResult single_result = searchQuery(query_data[q], k, ef);
            all_ids[q] = std::move(single_result.ids);
            all_distances[q] = std::move(single_result.distances);
        }
    }

    // Flatten results
    for (size_t q = 0; q < num_queries; ++q)
    {
        final_result.ids.insert(final_result.ids.end(), all_ids[q].begin(), all_ids[q].end());
        final_result.distances.insert(final_result.distances.end(), all_distances[q].begin(), all_distances[q].end());
    }

    return final_result;
}

/**
 * @brief Search single query across all layers (HNSW-style multi-layer search)
 * @param query Single query vector
 * @param k Number of nearest neighbors
 * @param ef Search parameter
 * @return SearchResult for this single query
 */
GannHNSW::SearchResult GannHNSW::searchQuery(const std::vector<float> &query, size_t k, size_t ef) const
{
    if (layers_.empty() || entry_point_ >= num_elements_)
    {
        return SearchResult{};
    }

    VertexId current_entry = entry_point_;

    // Search from top layer down to layer 1 (with ef=1 for efficiency)
    for (int level = static_cast<int>(layers_.size()) - 1; level > 0; --level)
    {
        if (static_cast<size_t>(level) < layers_.size() && !layers_[level].vertices.empty())
        {
            SearchResult layer_result = searchLayer(query, 1, 1, level, current_entry);
            if (!layer_result.ids.empty())
            {
                current_entry = layer_result.ids[0];
            }
        }
    }

    // Search layer 0 with full ef and return k results
    return searchLayer(query, k, ef, 0, current_entry);
}

void GannHNSW::save(const std::string &index_file) const
{
    try
    {
        std::ofstream os(index_file, std::ios::binary);
        if (!os.is_open())
        {
            throw std::runtime_error("Cannot open file for writing: " + index_file);
        }

        cereal::BinaryOutputArchive archive(os);
        archive(*this);
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Failed to save index: " + std::string(e.what()));
    }
}

bool GannHNSW::load(const std::string &index_file)
{
    try
    {
        std::ifstream is(index_file, std::ios::binary);
        if (!is.is_open())
        {
            return false;
        }

        cereal::BinaryInputArchive archive(is);
        archive(*this);

        // Re-initialize mutable random state after loading
        level_generator_.seed(std::random_device{}());
        level_distribution_ = std::uniform_real_distribution<double>(0.0, 1.0);

        return true;
    }
    catch (const std::exception &e)
    {
        return false;
    }
}