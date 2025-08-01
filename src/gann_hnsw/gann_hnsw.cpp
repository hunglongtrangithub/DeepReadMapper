#include "gann_hnsw.hpp"

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

    std::cout << "Building layer " << level << " with " << vertices.size() << " vertices..." << std::endl;

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
        vertex.neighbors.reserve(dmax_);
        vertex.explored = false;
        vertex.distance_to_query = 0.0f;
        layer.vertices.push_back(vertex);
    }

    // Phase 1: Build local graphs in parallel (20-70%)
    std::cout << "Phase 1: Building local graphs in parallel..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    buildLocalGraphsParallel(vertices, level, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Local graph construction completed in " << duration << " ms." << std::endl;

    // Phase 2: Merge local graphs (70-100%)
    std::cout << "Phase 2: Merging local graphs..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    mergeLocalGraphs(level);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Graph merging completed in " << duration << " ms." << std::endl;
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

    // Partition vertices into DISJOINT groups
    std::vector<std::vector<VertexId>> partitions(num_threads);
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        partitions[i % num_threads].push_back(vertices[i]);
    }

    // Each thread builds COMPLETE local NSW graphs (Algorithm 2, lines 2-8)
    std::vector<std::vector<Vertex>> local_graphs(num_threads);

    // Set the number of threads for OpenMP
    omp_set_num_threads(static_cast<int>(num_threads));

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        // Build COMPLETE local NSW graph for this partition
        buildLocalGraph(partitions[thread_id], local_graphs[thread_id], level);
    }

    // Store all local graphs for iterative merging
    Layer &layer = layers_[level];
    layer.vertices.clear();

    // Keep track of which vertices belong to which local graph for merging
    local_graph_partitions_.clear();
    local_graph_partitions_.resize(num_threads);

    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t start_idx = layer.vertices.size();
        layer.vertices.insert(layer.vertices.end(), local_graphs[t].begin(), local_graphs[t].end());
        size_t end_idx = layer.vertices.size();

        local_graph_partitions_[t] = {start_idx, end_idx};
    }
}

/**
 * @brief Build a complete local NSW graph for a partition (GANN Algorithm 2 Phase 1)
 */
void GannHNSW::buildLocalGraph(const std::vector<VertexId> &partition, std::vector<Vertex> &local_graph, size_t level)
{
    local_graph.clear();
    local_graph.reserve(partition.size());

    // Initialize all vertices in partition first
    for (VertexId vid : partition)
    {
        Vertex vertex;
        vertex.id = vid;
        vertex.explored = false;
        vertex.distance_to_query = 0.0f;
        vertex.neighbors.reserve(dmax_);
        local_graph.push_back(vertex);
    }

    // Build NSW graph: each vertex connects to its dmin_ nearest neighbors in the SAME partition
    for (size_t i = 0; i < local_graph.size(); ++i)
    {
        Vertex &current = local_graph[i];

        // Find dmin_ nearest neighbors within this partition
        std::vector<std::pair<Distance, size_t>> neighbor_candidates;

        for (size_t j = 0; j < local_graph.size(); ++j)
        {
            if (i == j)
                continue;

            Distance dist = computeDistance(data_[current.id], data_[local_graph[j].id]);
            neighbor_candidates.emplace_back(dist, j);
        }

        // Sort and take dmin_ closest
        std::sort(neighbor_candidates.begin(), neighbor_candidates.end());
        size_t num_neighbors = std::min(static_cast<size_t>(dmin_), neighbor_candidates.size());

        // Add forward edges
        for (size_t k = 0; k < num_neighbors; ++k)
        {
            VertexId neighbor_id = local_graph[neighbor_candidates[k].second].id;
            current.neighbors.push_back(neighbor_id);
        }

        // Add backward edges with CPU optimization (batch processing)
        for (size_t k = 0; k < num_neighbors; ++k)
        {
            size_t neighbor_idx = neighbor_candidates[k].second;
            local_graph[neighbor_idx].neighbors.push_back(current.id);
        }
    }

// Prune all vertices in batch to maintain dmax_
#pragma omp parallel for if (local_graph.size() > 100)
    for (size_t i = 0; i < local_graph.size(); ++i)
    {
        if (local_graph[i].neighbors.size() > dmax_)
        {
            pruneConnections(local_graph[i]);
        }
    }
}

/**
 * @brief Search within a local graph (simplified beam search for local construction)
 */
GannHNSW::SearchResult GannHNSW::searchLocalGraph(const std::vector<float> &query, const std::vector<Vertex> &local_graph, size_t k) const
{
    SearchResult result;
    if (local_graph.empty())
    {
        return result;
    }

    // For small local graphs, use simplified approach
    if (local_graph.size() <= dmin_)
    {
        // If local graph is small, just return all vertices sorted by distance
        std::vector<std::pair<Distance, VertexId>> candidates;

        for (const auto &vertex : local_graph)
        {
            Distance dist = computeDistance(query, data_[vertex.id]);
            candidates.emplace_back(dist, vertex.id);
        }

        std::sort(candidates.begin(), candidates.end());

        size_t return_count = std::min(k, candidates.size());
        for (size_t i = 0; i < return_count; ++i)
        {
            result.ids.push_back(candidates[i].second);
            result.distances.push_back(candidates[i].first);
        }

        return result;
    }

    // For larger local graphs, use beam search
    std::vector<Vertex> candidates;
    std::vector<bool> visited(local_graph.size(), false);

    // Start with first vertex as entry point
    candidates.push_back(local_graph[0]);
    candidates[0].distance_to_query = computeDistance(query, data_[local_graph[0].id]);
    visited[0] = true;

    // Beam search within local graph
    while (!candidates.empty())
    {
        // Sort candidates by distance
        std::sort(candidates.begin(), candidates.end(),
                  [](const Vertex &a, const Vertex &b)
                  {
                      return a.distance_to_query < b.distance_to_query;
                  });

        // Take closest candidate
        Vertex current = candidates[0];
        candidates.erase(candidates.begin());

        // Add to results if within top k
        if (result.ids.size() < k)
        {
            result.ids.push_back(current.id);
            result.distances.push_back(current.distance_to_query);
        }

        // Explore neighbors
        for (VertexId neighbor_id : current.neighbors)
        {
            // Find neighbor index in local graph
            auto neighbor_it = std::find_if(local_graph.begin(), local_graph.end(),
                                            [neighbor_id](const Vertex &v)
                                            {
                                                return v.id == neighbor_id;
                                            });

            if (neighbor_it != local_graph.end())
            {
                size_t neighbor_idx = std::distance(local_graph.begin(), neighbor_it);

                if (!visited[neighbor_idx])
                {
                    visited[neighbor_idx] = true;

                    Vertex neighbor_vertex = *neighbor_it;
                    neighbor_vertex.distance_to_query = computeDistance(query, data_[neighbor_id]);

                    candidates.push_back(neighbor_vertex);
                }
            }
        }

        // Limit candidate list size for efficiency
        if (candidates.size() > ef_construction_)
        {
            std::sort(candidates.begin(), candidates.end(),
                      [](const Vertex &a, const Vertex &b)
                      {
                          return a.distance_to_query < b.distance_to_query;
                      });
            candidates.resize(ef_construction_);
        }
    }

    return result;
}

/**
 * @brief Prune connections to maintain dmax_ limit using distance-based selection
 */
void GannHNSW::pruneConnections(Vertex &vertex)
{
    if (vertex.neighbors.size() <= dmax_)
        return;

    // Use the optimized computeDistance with AVX/AVX512
    std::vector<std::pair<Distance, VertexId>> neighbor_distances;
    neighbor_distances.reserve(vertex.neighbors.size());

    const std::vector<float> &vertex_data = data_[vertex.id];

    // CPU optimization: batch distance computation
    for (VertexId neighbor_id : vertex.neighbors)
    {
        Distance dist = computeDistance(vertex_data, data_[neighbor_id]);
        neighbor_distances.emplace_back(dist, neighbor_id);
    }

    // Partial sort for better CPU cache performance
    std::nth_element(neighbor_distances.begin(),
                     neighbor_distances.begin() + dmax_,
                     neighbor_distances.end());

    vertex.neighbors.clear();
    vertex.neighbors.reserve(dmax_);

    for (size_t i = 0; i < dmax_; ++i)
    {
        vertex.neighbors.push_back(neighbor_distances[i].second);
    }
}

/**
 * @brief Search in the global merged graph during construction (GANN Algorithm 2, line 12)
 * @param query Query vector
 * @param k Number of nearest neighbors to find
 * @param level Layer level to search in
 * @param entry_point Starting vertex (can be 0 for construction)
 * @return SearchResult containing found neighbors from the global graph
 */
GannHNSW::SearchResult GannHNSW::searchLayerForConstruction(const std::vector<float> &query,
                                                            size_t k,
                                                            size_t level,
                                                            VertexId entry_point) const
{
    SearchResult result;
    if (level >= layers_.size())
        return result;

    const Layer &layer = layers_[level];
    if (layer.vertices.empty())
        return result;

    // For construction, we search in the GLOBAL merged graph

    std::vector<std::pair<Distance, VertexId>> all_candidates;

    // Consider ALL vertices in the current layer (merged from all partitions)
    for (const auto &vertex : layer.vertices)
    {
        Distance dist = computeDistance(query, data_[vertex.id]);
        all_candidates.emplace_back(dist, vertex.id);
    }

    // Sort by distance and return top k
    std::sort(all_candidates.begin(), all_candidates.end());

    size_t return_count = std::min(k, all_candidates.size());
    result.ids.reserve(return_count);
    result.distances.reserve(return_count);

    for (size_t i = 0; i < return_count; ++i)
    {
        result.ids.push_back(all_candidates[i].second);
        result.distances.push_back(all_candidates[i].first);
    }

    return result;
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
    if (layer.vertices.empty() || local_graph_partitions_.empty())
        return;

    size_t num_partitions = local_graph_partitions_.size();

    if (num_partitions <= 1)
        return; // No merging needed

    std::cout << "Merging " << num_partitions << " local graphs..." << std::endl;

    // Hide cursor and create progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"Merging Graphs"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    auto start = std::chrono::high_resolution_clock::now();

    // GANN Algorithm 2: Iterative merging
    for (size_t iteration = 1; iteration < num_partitions; ++iteration)
    {
        std::vector<std::pair<VertexId, VertexId>> edge_list;
        std::vector<size_t> edge_indices;

// Phase 2a: Cross-graph search
#pragma omp parallel for
        for (size_t p = iteration; p < num_partitions; ++p)
        {
            auto &partition_range = local_graph_partitions_[p];

            for (size_t i = partition_range.first; i < partition_range.second; ++i)
            {
                Vertex &vertex = layer.vertices[i];

                SearchResult global_search = searchLayerForConstruction(
                    data_[vertex.id], dmin_, level, 0);

                std::vector<VertexId> local_neighbors = vertex.neighbors;
                vertex.neighbors.clear();
                vertex.neighbors.reserve(dmax_);

                std::set<VertexId> unique_neighbors;
                for (VertexId global_neighbor : global_search.ids)
                {
                    unique_neighbors.insert(global_neighbor);
                }
                for (VertexId local_neighbor : local_neighbors)
                {
                    unique_neighbors.insert(local_neighbor);
                }

                std::vector<std::pair<Distance, VertexId>> all_candidates;
                for (VertexId neighbor_id : unique_neighbors)
                {
                    Distance dist = computeDistance(data_[vertex.id], data_[neighbor_id]);
                    all_candidates.emplace_back(dist, neighbor_id);
                }

                std::sort(all_candidates.begin(), all_candidates.end());
                size_t max_neighbors = std::min(static_cast<size_t>(dmax_), all_candidates.size());

                for (size_t j = 0; j < max_neighbors; ++j)
                {
                    vertex.neighbors.push_back(all_candidates[j].second);
                }
            }
        }

        // Phase 2b: Collect backward edges
        for (const auto &vertex : layer.vertices)
        {
            for (VertexId neighbor_id : vertex.neighbors)
            {
                edge_list.emplace_back(neighbor_id, vertex.id);
            }
        }

        // Phase 2c: GatherScatter
        gatherScatter(edge_list, edge_indices);

        // Phase 2d: Update backward edges
        processBackwardEdges(layer, edge_list, edge_indices);

        // Update progress
        size_t progress_percent = (iteration * 100) / (num_partitions - 1);
        progressBar.set_progress(progress_percent);
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Graph merging completed in " << duration << " ms." << std::endl;
}

/**
 * @brief CPU-optimized GatherScatter operation (replaces GPU version)
 */
void GannHNSW::gatherScatter(std::vector<std::pair<VertexId, VertexId>> &edge_list,
                             std::vector<size_t> &indices)
{
    // Sort edges by target vertex for efficient grouping
    std::sort(edge_list.begin(), edge_list.end());

    // Build index array for CSR-like access
    indices.clear();
    indices.reserve(num_elements_ + 1);

    size_t current_vertex = 0;
    indices.push_back(0);

    for (size_t i = 0; i < edge_list.size(); ++i)
    {
        while (current_vertex < edge_list[i].first)
        {
            current_vertex++;
            indices.push_back(i);
        }
    }

    // Fill remaining indices
    while (indices.size() <= num_elements_)
    {
        indices.push_back(edge_list.size());
    }
}

/**
 * @brief CPU-optimized backward edge processing
 */
void GannHNSW::processBackwardEdges(Layer &layer,
                                    const std::vector<std::pair<VertexId, VertexId>> &edge_list,
                                    const std::vector<size_t> &indices)
{
#pragma omp parallel for
    for (size_t v = 0; v < layer.vertices.size(); ++v)
    {
        Vertex &vertex = layer.vertices[v];
        VertexId vertex_id = vertex.id;

        if (vertex_id >= indices.size() - 1)
            continue;

        // Collect all incoming edges for this vertex
        std::vector<std::pair<Distance, VertexId>> incoming_edges;

        for (size_t i = indices[vertex_id]; i < indices[vertex_id + 1]; ++i)
        {
            VertexId source_id = edge_list[i].second;
            Distance dist = computeDistance(data_[vertex_id], data_[source_id]);
            incoming_edges.emplace_back(dist, source_id);
        }

        // Merge with existing neighbors and prune to dmax_
        for (VertexId existing_neighbor : vertex.neighbors)
        {
            Distance dist = computeDistance(data_[vertex_id], data_[existing_neighbor]);
            incoming_edges.emplace_back(dist, existing_neighbor);
        }

        // Sort and keep best dmax_ neighbors
        std::sort(incoming_edges.begin(), incoming_edges.end());

        vertex.neighbors.clear();
        vertex.neighbors.reserve(dmax_);

        std::set<VertexId> unique_neighbors;
        for (const auto &edge : incoming_edges)
        {
            if (unique_neighbors.size() >= dmax_)
                break;
            if (unique_neighbors.insert(edge.second).second)
            {
                vertex.neighbors.push_back(edge.second);
            }
        }
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

    // Sequential computation with optimization in computeDistance()
    // For GPU, this could be replaced with a parallel kernel
    for (auto &candidate : candidates)
    {
        candidate.distance_to_query = computeDistance(query, data_[candidate.id]);
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

    std::cout << "Starting GANN-HNSW index construction for " << input_data.size() << " vectors..." << std::endl;

    // Hide cursor and create progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"Building Index"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    // Store data
    auto start = std::chrono::high_resolution_clock::now();
    data_ = input_data;
    num_elements_ = data_.size();
    progressBar.set_progress(10); // Data loading complete

    // Comment out to test on GANN-NSW first: single-layer HNSW
    // // Generate levels for all vertices following HNSW paper
    // std::vector<std::vector<VertexId>> level_vertices(max_level_ + 1);
    // VertexId highest_level_vertex = 0;
    // size_t highest_level = 0;

    // for (VertexId vid = 0; vid < num_elements_; ++vid)
    // {
    //     size_t vertex_level = getRandomLevel();

    //     // Track highest level vertex for entry point
    //     if (vertex_level > highest_level)
    //     {
    //         highest_level = vertex_level;
    //         highest_level_vertex = vid;
    //     }

    //     // Add vertex to all levels from 0 to vertex_level
    //     for (size_t level = 0; level <= vertex_level; ++level)
    //     {
    //         level_vertices[level].push_back(vid);
    //     }
    // }

    // // Set entry point to highest level vertex
    // entry_point_ = highest_level_vertex;

    // // Build each layer using GANN's Algorithm 2
    // for (size_t level = 0; level < level_vertices.size(); ++level)
    // {
    //     if (!level_vertices[level].empty())
    //     {
    //         buildLayer(level, level_vertices[level], num_threads);
    //     }
    // }

    std::vector<VertexId> all_vertices(num_elements_);
    for (VertexId vid = 0; vid < num_elements_; ++vid)
    {
        all_vertices[vid] = vid;
    }
    progressBar.set_progress(20); // Vertex initialization complete

    // Build single layer (Phase 1: 20-100%)
    buildLayer(0, all_vertices, num_threads);

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Index construction completed in " << duration << " ms." << std::endl;
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
    std::cout << "Starting search for " << num_queries << " queries (k=" << k << ", ef=" << ef << ")..." << std::endl;

    // Hide cursor and create progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"Searching"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    // Prepare result containers
    std::vector<std::vector<VertexId>> all_ids(num_queries);
    std::vector<std::vector<Distance>> all_distances(num_queries);

    auto start = std::chrono::high_resolution_clock::now();

    // Process queries in parallel using OpenMP
    omp_set_num_threads(static_cast<int>(num_threads));

    // Track progress with atomic counter
    std::atomic<size_t> completed_queries{0};

#pragma omp parallel for schedule(dynamic)
    for (size_t q = 0; q < num_queries; ++q)
    {
        SearchResult single_result = searchQuery(query_data[q], k, ef);
        all_ids[q] = std::move(single_result.ids);
        all_distances[q] = std::move(single_result.distances);

        // Update progress (thread-safe)
        size_t current_completed = completed_queries.fetch_add(1) + 1;
        size_t progress_percent = (current_completed * 100) / num_queries;

#pragma omp critical
        {
            progressBar.set_progress(progress_percent);
        }
    }

    // Flatten results
    for (size_t q = 0; q < num_queries; ++q)
    {
        final_result.ids.insert(final_result.ids.end(), all_ids[q].begin(), all_ids[q].end());
        final_result.distances.insert(final_result.distances.end(), all_distances[q].begin(), all_distances[q].end());
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Search completed in " << duration << " ms." << std::endl;

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

    // Search from top layer down to layer 1
    for (int level = static_cast<int>(layers_.size()) - 1; level > 0; --level)
    {
        std::cout << "Searching layer " << level << std::endl;
        if (static_cast<size_t>(level) < layers_.size() && !layers_[level].vertices.empty())
        {
            SearchResult layer_result = searchLayer(query, 1, ef, level, current_entry);
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