#include "post_processor.hpp"

//! Only use for special HNSW index with encoded labels. Ignore if using default sequential labels.
PositionInfo position_decode(size_t label, const std::vector<size_t> &mapping)
{
    // First, check if label exists in mapping
    if (label >= mapping.size() || label < 0)
    {
        throw std::out_of_range("Label " + std::to_string(label) + " out of range for mapping of size " + std::to_string(mapping.size()));
    };

    // Then, get the encoded label
    size_t mapped_label = mapping[label];

    // Decode position and strand info
    return {
        mapped_label >> 1,      // Extract position by right-shifting 1 bit
        (mapped_label & 1) != 0 // Extract RC flag from lowest bit
    };
}

// Template function to reformat hnsw output
template <typename NeighborType>
std::pair<std::vector<size_t>, std::vector<float>> reformat_output(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    size_t k)
{
    size_t n_rows = neighbors.size();

    // Flatten the 2D vectors into 1D arrays
    std::vector<size_t> host_indices(n_rows * k);
    std::vector<float> host_distances(n_rows * k);

    for (size_t i = 0; i < n_rows; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            host_indices[i * k + j] = static_cast<size_t>(neighbors[i][j]);
            host_distances[i * k + j] = distances[i][j];
        }
    }

    return {host_indices, host_distances};
}

std::string find_sequence(const std::string &ref_genome, size_t id, size_t ref_len)
{
    // 1 position creates 2 windows (forward and reverse complement)
    size_t genomic_pos = id / 2;
    int is_reverse = (id % 2 == 1);

    if (genomic_pos + ref_len > ref_genome.size())
    {
        return "";
    }

    std::string window = ref_genome.substr(genomic_pos, ref_len);

    if (is_reverse)
    {
        window = reverse_complement(window);
    }

    return window;
}

std::string find_sequence_static(const std::vector<std::string> &ref_seqs, size_t id)
{
    return ref_seqs[id];
}

// Dynamic lookup version - returns sequences, their dense IDs, and mapping indices to the unique pool
std::tuple<std::vector<std::string>, std::vector<size_t>, std::vector<size_t>> find_sequences(
    const std::string &ref_genome,
    const std::vector<size_t> &sparse_ids,
    size_t ref_len,
    size_t stride)
{
    if (sparse_ids.empty())
    {
        return {{}, {}, {}};
    }

    // Dense index (stride == 1): direct lookup, no expansion
    if (stride == 1)
    {
        std::vector<std::string> results;
        std::vector<size_t> expanded_ids;
        std::vector<size_t> mapping_ids;

        results.reserve(sparse_ids.size());
        expanded_ids.reserve(sparse_ids.size());
        mapping_ids.reserve(sparse_ids.size());

        for (size_t i = 0; i < sparse_ids.size(); ++i)
        {
            size_t id = sparse_ids[i];
            results.push_back(find_sequence(ref_genome, id, ref_len));
            expanded_ids.push_back(id);
            mapping_ids.push_back(i); // Direct 1-to-1 mapping
        }
        return {results, expanded_ids, mapping_ids};
    }

    // Sparse index (stride > 1): expand with deduplication
    std::cout << "[FIND_SEQUENCES] Processing " << sparse_ids.size() << " sparse IDs with stride=" << stride << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Step 1: Collect all expanded dense IDs with deduplication
    std::unordered_set<size_t> dense_id_set;
    std::vector<size_t> expansion_order; // Track order of first appearance
    size_t total_expansions = 0;

    for (size_t sparse_id : sparse_ids)
    {
        size_t actual_position = sparse_id * stride;

        if (actual_position >= ref_genome.size())
        {
            continue;
        }

        size_t start = (actual_position >= stride - 1) ? actual_position - stride + 1 : 0;
        size_t end = std::min(actual_position + stride, ref_genome.size());

        for (size_t pos = start; pos < end; ++pos)
        {
            auto [iter, inserted] = dense_id_set.insert(pos);
            if (inserted)
            {
                expansion_order.push_back(pos);
            }
            total_expansions++;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "[FIND_SEQUENCES] Expansion analysis:" << std::endl;
    std::cout << "  - Total expansions: " << total_expansions << std::endl;
    std::cout << "  - Unique dense IDs: " << expansion_order.size() << std::endl;
    std::cout << "  - Duplicates removed: " << (total_expansions - expansion_order.size()) << std::endl;
    std::cout << "  - Compression ratio: " << std::fixed << std::setprecision(2)
              << (100.0 * expansion_order.size() / total_expansions) << "%" << std::endl;
    std::cout << "  - Expansion time: " << duration.count() << " ms" << std::endl;

    // Step 2: Create mapping from dense_id to its index in the unique pool
    start_time = std::chrono::high_resolution_clock::now();
    std::unordered_map<size_t, size_t> dense_id_to_idx;
    for (size_t i = 0; i < expansion_order.size(); ++i)
    {
        dense_id_to_idx[expansion_order[i]] = i;
    }

    // Step 3: Fetch unique sequences using dynamic lookup
    std::vector<std::string> unique_seqs;
    unique_seqs.reserve(expansion_order.size());

    for (size_t pos : expansion_order)
    {
        unique_seqs.push_back(find_sequence(ref_genome, pos, ref_len));
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "  - Lookup time: " << duration.count() << " ms" << std::endl;

    // Step 4: Build mapping indices for all original expansions
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<size_t> mapping_ids;
    mapping_ids.reserve(total_expansions);

    for (size_t sparse_id : sparse_ids)
    {
        size_t actual_position = sparse_id * stride;

        if (actual_position >= ref_genome.size())
        {
            continue;
        }

        size_t start = (actual_position >= stride - 1) ? actual_position - stride + 1 : 0;
        size_t end = std::min(actual_position + stride, ref_genome.size());

        for (size_t pos = start; pos < end; ++pos)
        {
            // Map this dense_id to its index in unique pool
            mapping_ids.push_back(dense_id_to_idx[pos]);
        }
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "  - Mapping creation: " << duration.count() << " ms" << std::endl;
    std::cout << "  - Final sequences: " << unique_seqs.size() << std::endl;
    std::cout << "  - Mapping indices: " << mapping_ids.size() << std::endl;

    return {unique_seqs, expansion_order, mapping_ids};
}

// Static lookup version - returns sequences, their dense IDs, and mapping indices to the unique pool
std::tuple<std::vector<std::string>, std::vector<size_t>, std::vector<size_t>> find_sequences(
    const std::vector<std::string> &ref_seqs,
    const std::vector<size_t> &sparse_ids,
    size_t stride)
{
    if (sparse_ids.empty())
    {
        return {{}, {}, {}};
    }

    // Dense index (stride == 1): direct lookup, no expansion
    if (stride == 1)
    {
        std::vector<std::string> results;
        std::vector<size_t> expanded_ids;
        std::vector<size_t> mapping_ids;

        results.reserve(sparse_ids.size());
        expanded_ids.reserve(sparse_ids.size());
        mapping_ids.reserve(sparse_ids.size());

        for (size_t i = 0; i < sparse_ids.size(); ++i)
        {
            size_t id = sparse_ids[i];
            if (id < ref_seqs.size())
            {
                results.push_back(ref_seqs[id]);
                expanded_ids.push_back(id);
                mapping_ids.push_back(i); // Direct 1-to-1 mapping
            }
        }
        return {results, expanded_ids, mapping_ids};
    }

    // Sparse index (stride > 1): expand with deduplication
    std::cout << "[FIND_SEQUENCES] Processing " << sparse_ids.size() << " sparse IDs with stride=" << stride << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Step 1: Collect all expanded dense IDs with deduplication
    std::unordered_set<size_t> dense_id_set;
    std::vector<size_t> expansion_order; // Track order of first appearance
    size_t total_expansions = 0;

    for (size_t sparse_id : sparse_ids)
    {
        size_t actual_position = sparse_id * stride;

        if (actual_position >= ref_seqs.size())
        {
            continue;
        }

        size_t start = (actual_position >= stride - 1) ? actual_position - stride + 1 : 0;
        size_t end = std::min(actual_position + stride, ref_seqs.size());

        for (size_t pos = start; pos < end; ++pos)
        {
            auto [iter, inserted] = dense_id_set.insert(pos);
            if (inserted)
            {
                expansion_order.push_back(pos);
            }
            total_expansions++;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "[FIND_SEQUENCES] Expansion analysis:" << std::endl;
    std::cout << "  - Total expansions: " << total_expansions << std::endl;
    std::cout << "  - Unique dense IDs: " << expansion_order.size() << std::endl;
    std::cout << "  - Duplicates removed: " << (total_expansions - expansion_order.size()) << std::endl;
    std::cout << "  - Compression ratio: " << std::fixed << std::setprecision(2)
              << (100.0 * expansion_order.size() / total_expansions) << "%" << std::endl;
    std::cout << "  - Expansion time: " << duration.count() << " ms" << std::endl;

    // Step 2: Create mapping from dense_id to its index in the unique pool
    start_time = std::chrono::high_resolution_clock::now();
    std::unordered_map<size_t, size_t> dense_id_to_idx;
    for (size_t i = 0; i < expansion_order.size(); ++i)
    {
        dense_id_to_idx[expansion_order[i]] = i;
    }

    // Step 3: Fetch unique sequences
    std::vector<std::string> unique_seqs;
    unique_seqs.reserve(expansion_order.size());

    for (size_t pos : expansion_order)
    {
        if (pos < ref_seqs.size())
        {
            unique_seqs.push_back(ref_seqs[pos]);
        }
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "  - Lookup time: " << duration.count() << " ms" << std::endl;

    // Step 4: Build mapping indices for all original expansions
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<size_t> mapping_ids;
    mapping_ids.reserve(total_expansions);

    for (size_t sparse_id : sparse_ids)
    {
        size_t actual_position = sparse_id * stride;

        if (actual_position >= ref_seqs.size())
        {
            continue;
        }

        size_t start = (actual_position >= stride - 1) ? actual_position - stride + 1 : 0;
        size_t end = std::min(actual_position + stride, ref_seqs.size());

        for (size_t pos = start; pos < end; ++pos)
        {
            // Map this dense_id to its index in unique pool
            mapping_ids.push_back(dense_id_to_idx[pos]);
        }
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "  - Mapping creation: " << duration.count() << " ms" << std::endl;
    std::cout << "  - Final sequences: " << unique_seqs.size() << std::endl;
    std::cout << "  - Mapping indices: " << mapping_ids.size() << std::endl;

    return {unique_seqs, expansion_order, mapping_ids};
}

// Helper function to convert neighbor types to size_t
template <typename NeighborType>
std::vector<std::vector<size_t>> convert_neighbors(const std::vector<std::vector<NeighborType>> &neighbors)
{
    if constexpr (std::is_same_v<NeighborType, size_t>)
    {
        return neighbors;
    }
    else
    {
        std::vector<std::vector<size_t>> converted(neighbors.size());
        for (size_t i = 0; i < neighbors.size(); ++i)
        {
            converted[i].assign(neighbors[i].begin(), neighbors[i].end());
        }
        return converted;
    }
}

template <typename NeighborType>
std::tuple<std::vector<std::string>, std::vector<int>, std::vector<size_t>> post_process_sw_dynamic(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::string &ref_genome,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k, size_t k_clusters)
{
    std::cout << "[POST-PROCESS] Using dynamic lookup & SW reranker" << std::endl;
    std::cout << "[POST-PROCESS] Configs:" << std::endl;
    std::cout << "[POST-PROCESS] Metrics: sw_score" << std::endl;
    std::cout << "[POST-PROCESS] Neighbors type: " << typeid(NeighborType).name() << std::endl;

    auto converted_neighbors = convert_neighbors(neighbors);
    size_t total_queries = query_seqs.size();

    // Pre-allocate results for each query
    std::vector<std::vector<std::string>> query_final_seqs(total_queries);
    std::vector<std::vector<int>> query_scores(total_queries);
    std::vector<std::vector<size_t>> all_ids(total_queries);

    // Thread-safe progress tracking
    std::atomic<size_t> completed_queries(0);

    // Hide cursor and create progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"post-processing"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    if (k > k_clusters * 2 * stride)
    {
        throw std::runtime_error("Final k too large. Ensure k < k_clusters * 2 * stride to have enough candidates.");
    }

#pragma omp parallel for num_threads(Config::Postprocess::NUM_THREADS) schedule(dynamic)
    for (size_t i = 0; i < total_queries; ++i)
    {
        size_t num_cands_per_query = std::min(k_clusters, converted_neighbors[i].size());
        std::vector<size_t> query_neighbors(converted_neighbors[i].begin(), converted_neighbors[i].begin() + num_cands_per_query);

        // Get candidate sequences using dynamic lookup (returns unique seqs + mapping)
        auto [unique_seqs, unique_ids, mapping_to_unique] = find_sequences(ref_genome, query_neighbors, ref_len, stride);

        // Expand sequences back to original count using mapping
        std::vector<std::string> query_cand_seqs;
        std::vector<size_t> expanded_ids;
        query_cand_seqs.reserve(mapping_to_unique.size());
        expanded_ids.reserve(mapping_to_unique.size());

        for (size_t idx : mapping_to_unique)
        {
            query_cand_seqs.push_back(unique_seqs[idx]);
            expanded_ids.push_back(unique_ids[idx]);
        }

        // Rerank candidates using Smith-Waterman
        auto [top_seqs, top_scores, top_ids] = sw_reranker(query_cand_seqs, expanded_ids, query_seqs[i], k);

        // Store results for this query
        query_final_seqs[i] = std::move(top_seqs);
        query_scores[i] = std::move(top_scores);
        all_ids[i] = std::move(top_ids);

        // Update progress bar (thread-safe)
        size_t current_completed = completed_queries.fetch_add(1) + 1;
        if (current_completed % 100 == 0)
        {
            size_t current_progress_percent = (current_completed * 100) / total_queries;
            progressBar.set_progress(current_progress_percent);
        }
    }

    // Complete progress bar and show cursor
    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    // Flatten results
    std::vector<std::string> final_seqs;
    std::vector<int> scores;
    std::vector<size_t> final_ids;
    final_seqs.reserve(total_queries * k);
    scores.reserve(total_queries * k);
    final_ids.reserve(total_queries * k);

    for (size_t i = 0; i < total_queries; ++i)
    {
        final_seqs.insert(final_seqs.end(), query_final_seqs[i].begin(), query_final_seqs[i].end());
        scores.insert(scores.end(), query_scores[i].begin(), query_scores[i].end());
        final_ids.insert(final_ids.end(), all_ids[i].begin(), all_ids[i].end());
    }

    return {final_seqs, scores, final_ids};
}

template <typename NeighborType>
std::tuple<std::vector<std::string>, std::vector<int>, std::vector<size_t>> post_process_sw_static(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::vector<std::string> &ref_seqs,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k, size_t k_clusters)
{
    std::cout << "[POST-PROCESS] Using static lookup & SW reranker" << std::endl;
    std::cout << "[POST-PROCESS] Configs:" << std::endl;
    std::cout << "[POST-PROCESS] Metrics: sw_score" << std::endl;
    std::cout << "[POST-PROCESS] Neighbors type: " << typeid(NeighborType).name() << std::endl;

    auto converted_neighbors = convert_neighbors(neighbors);
    size_t total_queries = query_seqs.size();

    // Pre-allocate results for each query
    std::vector<std::vector<std::string>> query_final_seqs(total_queries);
    std::vector<std::vector<int>> query_scores(total_queries);
    std::vector<std::vector<size_t>> all_ids(total_queries);

    // Thread-safe progress tracking
    std::atomic<size_t> completed_queries(0);

    // Hide cursor and create progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"post-processing"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    if (k > k_clusters * 2 * stride)
    {
        throw std::runtime_error("Final k too large. Ensure k < k_clusters * 2 * stride to have enough candidates.");
    }

#pragma omp parallel for num_threads(Config::Postprocess::NUM_THREADS) schedule(dynamic)
    for (size_t i = 0; i < total_queries; ++i)
    {
        size_t num_cands_per_query = std::min(k_clusters, converted_neighbors[i].size());
        std::vector<size_t> query_neighbors(converted_neighbors[i].begin(), converted_neighbors[i].begin() + num_cands_per_query);

        // Get candidate sequences using static lookup (returns unique seqs + mapping)
        auto [unique_seqs, unique_ids, mapping_to_unique] = find_sequences(ref_seqs, query_neighbors, stride);

        // Expand sequences back to original count using mapping
        std::vector<std::string> query_cand_seqs;
        std::vector<size_t> expanded_ids;
        query_cand_seqs.reserve(mapping_to_unique.size());
        expanded_ids.reserve(mapping_to_unique.size());

        for (size_t idx : mapping_to_unique)
        {
            query_cand_seqs.push_back(unique_seqs[idx]);
            expanded_ids.push_back(unique_ids[idx]);
        }

        // Rerank candidates using Smith-Waterman
        auto [top_seqs, top_scores, top_ids] = sw_reranker(query_cand_seqs, expanded_ids, query_seqs[i], k);

        // Store results for this query
        query_final_seqs[i] = std::move(top_seqs);
        query_scores[i] = std::move(top_scores);
        all_ids[i] = std::move(top_ids);

        // Update progress bar (thread-safe)
        size_t current_completed = completed_queries.fetch_add(1) + 1;
        if (current_completed % 100 == 0)
        {
            size_t current_progress_percent = (current_completed * 100) / total_queries;
            progressBar.set_progress(current_progress_percent);
        }
    }

    // Complete progress bar and show cursor
    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    // Flatten results
    std::vector<std::string> final_seqs;
    std::vector<int> scores;
    std::vector<size_t> final_ids;
    final_seqs.reserve(total_queries * k);
    scores.reserve(total_queries * k);
    final_ids.reserve(total_queries * k);

    for (size_t i = 0; i < total_queries; ++i)
    {
        final_seqs.insert(final_seqs.end(), query_final_seqs[i].begin(), query_final_seqs[i].end());
        scores.insert(scores.end(), query_scores[i].begin(), query_scores[i].end());
        final_ids.insert(final_ids.end(), all_ids[i].begin(), all_ids[i].end());
    }

    return {final_seqs, scores, final_ids};
}

// L2 distance version with dynamic lookup (templated)
template <typename NeighborType>
std::tuple<std::vector<std::string>, std::vector<float>, std::vector<size_t>> post_process_l2_dynamic(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::string &ref_genome,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k,
    const std::vector<std::vector<float>> &query_embeddings,
    Vectorizer &vectorizer, size_t k_clusters)
{
    std::cout << "[POST-PROCESS] Using dynamic lookup & batched reranker" << std::endl;
    std::cout << "[POST-PROCESS] Configs:" << std::endl;
    std::cout << "[POST-PROCESS] Metrics: L2 distance" << std::endl;
    std::cout << "[POST-PROCESS] Neighbors type: " << typeid(NeighborType).name() << std::endl;

    if (k > k_clusters * 2 * stride && stride > 1)
    {
        throw std::runtime_error("Final k too large. Ensure k < k_clusters * 2 * stride to have enough candidates.");
    }

    auto converted_neighbors = convert_neighbors(neighbors);
    size_t total_queries = query_embeddings.size();

    // Step 1: Flatten sparse IDs
    std::cout << "[POST-PROCESS] Flattening candidates for " << total_queries << " queries." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<size_t> sparse_ids;
    std::vector<size_t> query_start_indices_sparse;

    for (size_t i = 0; i < total_queries; ++i)
    {
        query_start_indices_sparse.push_back(sparse_ids.size());
        size_t num_cands_per_query = (stride == 1)
                                         ? std::min(k, converted_neighbors[i].size())
                                         : std::min(k_clusters, converted_neighbors[i].size());
        sparse_ids.insert(sparse_ids.end(),
                          converted_neighbors[i].begin(),
                          converted_neighbors[i].begin() + num_cands_per_query);
    }
    query_start_indices_sparse.push_back(sparse_ids.size());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Flattening completed in " << duration.count() << " ms" << std::endl;

    // Step 2: Get unique sequences + mapping (using dynamic lookup)
    std::cout << "[POST-PROCESS] Fetching candidate sequences with deduplication." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    auto [unique_seqs, unique_dense_ids, mapping_to_unique] = find_sequences(ref_genome, sparse_ids, ref_len, stride);

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Fetching completed in " << duration.count() << " ms" << std::endl;

    // Step 3: Build query boundaries in expanded space
    std::cout << "[POST-PROCESS] Building query boundaries." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    std::vector<size_t> query_start_indices_expanded;
    query_start_indices_expanded.reserve(total_queries + 1);
    query_start_indices_expanded.push_back(0);

    for (size_t q = 0; q < total_queries; ++q)
    {
        size_t sparse_start = query_start_indices_sparse[q];
        size_t sparse_end = query_start_indices_sparse[q + 1];
        size_t num_sparse = sparse_end - sparse_start;
        size_t expansions = num_sparse * (2 * stride - 1);
        query_start_indices_expanded.push_back(query_start_indices_expanded.back() + expansions);
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Boundary building completed in " << duration.count() << " ms" << std::endl;

    // Early termination for stride == 1
    if (stride == 1)
    {
        std::cout << "[POST-PROCESS] stride == 1 (dense), skipping reranker" << std::endl;

        std::vector<std::string> final_seqs;
        std::vector<float> final_scores;
        std::vector<size_t> final_ids;
        final_seqs.reserve(total_queries * k);
        final_scores.reserve(total_queries * k);
        final_ids.reserve(total_queries * k);

        for (size_t i = 0; i < total_queries; ++i)
        {
            size_t start_idx = query_start_indices_expanded[i];
            size_t end_idx = query_start_indices_expanded[i + 1];
            size_t actual_cands = std::min(k, end_idx - start_idx);

            for (size_t j = 0; j < actual_cands; ++j)
            {
                size_t unique_idx = mapping_to_unique[start_idx + j];
                final_seqs.push_back(unique_seqs[unique_idx]);
                final_scores.push_back(distances[i][j]);
                final_ids.push_back(unique_dense_ids[unique_idx]);
            }
        }

        return {final_seqs, final_scores, final_ids};
    }

    // Step 4: Vectorize ONLY unique sequences
    std::cout << "[POST-PROCESS] Vectorizing " << unique_seqs.size()
              << " unique candidates (deduplicated from " << mapping_to_unique.size()
              << " total expansions)." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    const size_t CHUNK_SIZE = Config::Postprocess::CHUNK_SIZE;
    std::vector<std::vector<float>> unique_embeddings;
    unique_embeddings.reserve(unique_seqs.size());

    for (size_t chunk_start = 0; chunk_start < unique_seqs.size(); chunk_start += CHUNK_SIZE)
    {
        size_t chunk_end = std::min(chunk_start + CHUNK_SIZE, unique_seqs.size());
        std::vector<std::string> chunk(unique_seqs.begin() + chunk_start,
                                       unique_seqs.begin() + chunk_end);
        std::vector<std::vector<float>> chunk_embeddings = vectorizer.vectorize(chunk, false);
        unique_embeddings.insert(unique_embeddings.end(),
                                 chunk_embeddings.begin(),
                                 chunk_embeddings.end());
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Vectorization completed in " << duration.count() << " ms" << std::endl;
    std::cout << "[POST-PROCESS] Saved " << (mapping_to_unique.size() - unique_seqs.size())
              << " redundant vectorizations!" << std::endl;

    // Step 5: Expand data using mapping
    std::cout << "[POST-PROCESS] Expanding data using mapping for batch reranker." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::string> expanded_seqs;
    std::vector<size_t> expanded_ids;
    std::vector<size_t> expand_embedding_ids;
    expanded_seqs.reserve(mapping_to_unique.size());
    expanded_ids.reserve(mapping_to_unique.size());
    expand_embedding_ids.reserve(mapping_to_unique.size());

    for (size_t unique_idx : mapping_to_unique)
    {
        expanded_seqs.push_back(unique_seqs[unique_idx]);
        expanded_ids.push_back(unique_dense_ids[unique_idx]);
        expand_embedding_ids.push_back(unique_idx);
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Expansion completed in " << duration.count() << " ms" << std::endl;

    // Step 6: Call batch_reranker with pre-computed embeddings
    std::cout << "[POST-PROCESS] Running batched reranker with pre-computed embeddings." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    auto batch_results = batch_reranker(
        expanded_seqs,
        expanded_ids,
        expand_embedding_ids,
        unique_embeddings,
        query_start_indices_expanded,
        query_embeddings,
        k);

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Reranking completed in " << duration.count() << " ms" << std::endl;

    // Step 7: Flatten results
    std::cout << "[POST-PROCESS] Formatting final results." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::string> final_seqs;
    std::vector<float> final_scores;
    std::vector<size_t> final_ids;
    final_seqs.reserve(total_queries * k);
    final_scores.reserve(total_queries * k);
    final_ids.reserve(total_queries * k);

    for (const auto &[seqs, scores, ids] : batch_results)
    {
        final_seqs.insert(final_seqs.end(), seqs.begin(), seqs.end());
        final_scores.insert(final_scores.end(), scores.begin(), scores.end());
        final_ids.insert(final_ids.end(), ids.begin(), ids.end());
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Result formatting completed in " << duration.count() << " ms" << std::endl;

    return {final_seqs, final_scores, final_ids};
}

// L2 distance version with dynamic lookup with streaming output (templated)
template <typename NeighborType>
void post_process_l2_dynamic_streaming(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::string &ref_genome,
    const std::vector<std::string> &query_seqs,
    const std::vector<std::string> &query_ids,
    size_t ref_len, size_t stride, size_t k,
    const std::vector<std::vector<float>> &query_embeddings,
    Vectorizer &vectorizer, size_t k_clusters,
    const std::string &sam_file,
    const std::string &ref_name)
{
    std::cout << "[POST-PROCESS] Using dynamic lookup & batched reranker with STREAMING output" << std::endl;
    std::cout << "[POST-PROCESS] Configs:" << std::endl;
    std::cout << "[POST-PROCESS] Metrics: L2 distance" << std::endl;
    std::cout << "[POST-PROCESS] Neighbors type: " << typeid(NeighborType).name() << std::endl;

    if (k > k_clusters * 2 * stride)
    {
        throw std::runtime_error("Final k too large. Ensure k < k_clusters * 2 * stride to have enough candidates.");
    }

    auto converted_neighbors = convert_neighbors(neighbors);
    size_t total_queries = query_embeddings.size();

    // ========== GLOBAL DEDUPLICATION (Outside batching) ==========

    // Step 1: Flatten ALL sparse IDs for ALL queries
    std::cout << "[POST-PROCESS] Flattening candidates for " << total_queries << " queries." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<size_t> sparse_ids;
    std::vector<size_t> query_start_indices_sparse;

    for (size_t i = 0; i < total_queries; ++i)
    {
        query_start_indices_sparse.push_back(sparse_ids.size());
        size_t num_cands_per_query = (stride == 1)
                                         ? std::min(k, converted_neighbors[i].size())
                                         : std::min(k_clusters, converted_neighbors[i].size());
        sparse_ids.insert(sparse_ids.end(),
                          converted_neighbors[i].begin(),
                          converted_neighbors[i].begin() + num_cands_per_query);
    }
    query_start_indices_sparse.push_back(sparse_ids.size());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Flattening completed in " << duration.count() << " ms" << std::endl;

    // Step 2: Get unique sequences + mapping (GLOBAL deduplication across ALL queries)
    std::cout << "[POST-PROCESS] Fetching candidate sequences with GLOBAL deduplication." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    auto [unique_seqs, unique_dense_ids, mapping_to_unique] = find_sequences(ref_genome, sparse_ids, ref_len, stride);

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Fetching completed in " << duration.count() << " ms" << std::endl;

    // Step 3: Build query boundaries in expanded space
    std::cout << "[POST-PROCESS] Building query boundaries." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    std::vector<size_t> query_start_indices_expanded;
    query_start_indices_expanded.reserve(total_queries + 1);
    query_start_indices_expanded.push_back(0);

    for (size_t q = 0; q < total_queries; ++q)
    {
        size_t sparse_start = query_start_indices_sparse[q];
        size_t sparse_end = query_start_indices_sparse[q + 1];
        size_t num_sparse = sparse_end - sparse_start;
        size_t expansions = num_sparse * (2 * stride - 1);
        query_start_indices_expanded.push_back(query_start_indices_expanded.back() + expansions);
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Boundary building completed in " << duration.count() << " ms" << std::endl;

    // Early termination for stride == 1 (dense index)
    if (stride == 1)
    {
        std::cout << "[POST-PROCESS] stride == 1 (dense), skipping reranker, streaming directly" << std::endl;

        // Stream in batches to avoid memory overflow
        const size_t QUERY_BATCH_SIZE = Config::Postprocess::QUERY_BATCH_SIZE;
        const size_t num_batches = (total_queries + QUERY_BATCH_SIZE - 1) / QUERY_BATCH_SIZE;

        indicators::show_console_cursor(false);
        indicators::ProgressBar streamBar{
            indicators::option::BarWidth{80},
            indicators::option::PrefixText{"streaming results"},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true}};

        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx)
        {
            size_t batch_start = batch_idx * QUERY_BATCH_SIZE;
            size_t batch_end = std::min(batch_start + QUERY_BATCH_SIZE, total_queries);

            std::vector<std::string> batch_seqs;
            std::vector<float> batch_scores;
            std::vector<size_t> batch_ids;

            for (size_t i = batch_start; i < batch_end; ++i)
            {
                size_t start_idx = query_start_indices_expanded[i];
                size_t end_idx = query_start_indices_expanded[i + 1];
                size_t actual_cands = std::min(k, end_idx - start_idx);

                for (size_t j = 0; j < actual_cands; ++j)
                {
                    size_t unique_idx = mapping_to_unique[start_idx + j];
                    batch_seqs.push_back(unique_seqs[unique_idx]);
                    batch_scores.push_back(distances[i][j]);
                    batch_ids.push_back(unique_dense_ids[unique_idx]);
                }
            }

            bool write_header = (batch_idx == 0);
            write_sam_streaming(batch_seqs, batch_scores, query_seqs, query_ids, batch_ids,
                                ref_name, ref_len, k, sam_file, batch_start, write_header);

            streamBar.set_progress(((batch_idx + 1) * 100) / num_batches);
        }

        streamBar.set_progress(100);
        indicators::show_console_cursor(true);
        std::cout << "\n[POST-PROCESS] Streaming completed!" << std::endl;
        return;
    }

    // ========== BATCHED VECTORIZATION & RERANKING (sparse index) ==========

    // Step 4: Vectorize ONLY unique sequences (in chunks to avoid memory overflow)
    std::cout << "[POST-PROCESS] Vectorizing " << unique_seqs.size()
              << " unique candidates (deduplicated from " << mapping_to_unique.size()
              << " total expansions)." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    const size_t CHUNK_SIZE = Config::Postprocess::CHUNK_SIZE;
    std::vector<std::vector<float>> unique_embeddings;
    unique_embeddings.reserve(unique_seqs.size());

    for (size_t chunk_start = 0; chunk_start < unique_seqs.size(); chunk_start += CHUNK_SIZE)
    {
        size_t chunk_end = std::min(chunk_start + CHUNK_SIZE, unique_seqs.size());
        std::vector<std::string> chunk(unique_seqs.begin() + chunk_start,
                                       unique_seqs.begin() + chunk_end);
        std::vector<std::vector<float>> chunk_embeddings = vectorizer.vectorize(chunk, false);
        unique_embeddings.insert(unique_embeddings.end(),
                                 chunk_embeddings.begin(),
                                 chunk_embeddings.end());
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Vectorization completed in " << duration.count() << " ms" << std::endl;
    std::cout << "[POST-PROCESS] Saved " << (mapping_to_unique.size() - unique_seqs.size())
              << " redundant vectorizations!" << std::endl;

    // Step 5: Batch reranking and streaming
    const size_t QUERY_BATCH_SIZE = Config::Postprocess::QUERY_BATCH_SIZE;
    const size_t num_batches = (total_queries + QUERY_BATCH_SIZE - 1) / QUERY_BATCH_SIZE;

    std::cout << "[POST-PROCESS] Reranking and streaming " << total_queries << " queries in "
              << num_batches << " batches (size: " << QUERY_BATCH_SIZE << ")" << std::endl;

    indicators::show_console_cursor(false);
    indicators::ProgressBar batchProgressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"reranking & streaming batches"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx)
    {
        size_t batch_start = batch_idx * QUERY_BATCH_SIZE;
        size_t batch_end = std::min(batch_start + QUERY_BATCH_SIZE, total_queries);
        size_t batch_size = batch_end - batch_start;

        // Build batch-specific query boundaries
        std::vector<size_t> batch_query_boundaries;
        batch_query_boundaries.reserve(batch_size + 1);
        batch_query_boundaries.push_back(0);

        size_t offset = 0;
        for (size_t i = batch_start; i < batch_end; ++i)
        {
            size_t num_cands = query_start_indices_expanded[i + 1] - query_start_indices_expanded[i];
            offset += num_cands;
            batch_query_boundaries.push_back(offset);
        }

        // Extract batch query embeddings
        std::vector<std::vector<float>> batch_query_embeddings(
            query_embeddings.begin() + batch_start,
            query_embeddings.begin() + batch_end);

        // Expand to get all candidates in this batch
        size_t batch_cand_start = query_start_indices_expanded[batch_start];
        size_t batch_cand_end = query_start_indices_expanded[batch_end];
        size_t batch_cand_count = batch_cand_end - batch_cand_start;

        std::vector<std::string> batch_seqs;
        std::vector<size_t> batch_ids;
        std::vector<size_t> batch_embedding_indices;
        batch_seqs.reserve(batch_cand_count);
        batch_ids.reserve(batch_cand_count);
        batch_embedding_indices.reserve(batch_cand_count);

        for (size_t i = batch_cand_start; i < batch_cand_end; ++i)
        {
            size_t unique_idx = mapping_to_unique[i];
            batch_seqs.push_back(unique_seqs[unique_idx]);
            batch_ids.push_back(unique_dense_ids[unique_idx]);
            batch_embedding_indices.push_back(unique_idx);
        }

        // Rerank batch
        auto batch_results = batch_reranker(
            batch_seqs,
            batch_ids,
            batch_embedding_indices,
            unique_embeddings,
            batch_query_boundaries,
            batch_query_embeddings,
            k);

        batch_seqs.clear();
        batch_seqs.shrink_to_fit();
        batch_ids.clear();
        batch_ids.shrink_to_fit();
        batch_embedding_indices.clear();
        batch_embedding_indices.shrink_to_fit();

        // Flatten and stream results
        std::vector<std::string> result_seqs;
        std::vector<float> result_scores;
        std::vector<size_t> result_ids;

        for (const auto &[seqs, scores, ids] : batch_results)
        {
            result_seqs.insert(result_seqs.end(), seqs.begin(), seqs.end());
            result_scores.insert(result_scores.end(), scores.begin(), scores.end());
            result_ids.insert(result_ids.end(), ids.begin(), ids.end());
        }

        bool write_header = (batch_idx == 0);
        write_sam_streaming(result_seqs, result_scores, query_seqs, query_ids, result_ids,
                            ref_name, ref_len, k, sam_file, batch_start, write_header);

        result_seqs.clear();
        result_seqs.shrink_to_fit();
        result_scores.clear();
        result_scores.shrink_to_fit();
        result_ids.clear();
        result_ids.shrink_to_fit();

        batchProgressBar.set_progress(((batch_idx + 1) * 100) / num_batches);
    }

    batchProgressBar.set_progress(100);
    indicators::show_console_cursor(true);

    std::cout << "\n[POST-PROCESS] All batches completed and streamed to disk!" << std::endl;
}

template <typename NeighborType>
std::tuple<std::vector<std::string>, std::vector<float>, std::vector<size_t>> post_process_l2_static(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::vector<std::string> &ref_seqs,
    const std::vector<std::string> &query_seqs,
    size_t ref_len,
    size_t stride,
    size_t k,
    const std::vector<std::vector<float>> &query_embeddings,
    Vectorizer &vectorizer,
    size_t k_clusters)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "[POST-PROCESS] Starting post-processing with L2 reranker (static lookup)" << std::endl;

    // Step 1: Collect all neighbor IDs
    std::vector<size_t> all_neighbor_ids;
    size_t total_neighbors = 0;
    for (const auto &neighbor_vec : neighbors)
    {
        total_neighbors += neighbor_vec.size();
    }
    all_neighbor_ids.reserve(total_neighbors);

    for (const auto &neighbor_vec : neighbors)
    {
        for (const auto &neighbor : neighbor_vec)
        {
            all_neighbor_ids.push_back(static_cast<size_t>(neighbor));
        }
    }

    std::cout << "[POST-PROCESS] Collected " << all_neighbor_ids.size() << " neighbor IDs from " << neighbors.size() << " queries" << std::endl;

    // Step 2: Fetch sequences with deduplication
    start_time = std::chrono::high_resolution_clock::now();
    auto [unique_seqs, dense_ids, mapping_ids] = find_sequences(ref_seqs, all_neighbor_ids, stride);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "[POST-PROCESS] Fetching completed in " << duration.count() << " ms" << std::endl;

    // Step 3: Build query boundaries
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<size_t> query_start_indices;
    query_start_indices.reserve(neighbors.size() + 1);
    size_t cumulative = 0;
    for (const auto &neighbor_vec : neighbors)
    {
        query_start_indices.push_back(cumulative);
        cumulative += neighbor_vec.size() * (stride > 1 ? stride : 1);
    }
    query_start_indices.push_back(cumulative);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Boundary building completed in " << duration.count() << " ms" << std::endl;

    // Step 4: Vectorize unique sequences
    start_time = std::chrono::high_resolution_clock::now();
    size_t total_expansions = mapping_ids.size();
    std::cout << "[POST-PROCESS] Vectorizing " << unique_seqs.size() << " unique candidates (deduplicated from " << total_expansions << " total expansions)." << std::endl;
    std::vector<std::vector<float>> unique_embeddings = vectorizer.vectorize(unique_seqs, false);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Vectorization completed in " << duration.count() << " ms" << std::endl;
    std::cout << "[POST-PROCESS] Saved " << (total_expansions - unique_seqs.size()) << " redundant vectorizations!" << std::endl;

    // Step 5: Expand embeddings and sequences using mapping_ids
    start_time = std::chrono::high_resolution_clock::now();
    std::cout << "[POST-PROCESS] Expanding sequences using mapping_ids." << std::endl;

    std::vector<std::string> cand_seqs;
    std::vector<size_t> expand_cand_ids;
    std::vector<size_t> cand_embedding_ids;

    cand_seqs.reserve(mapping_ids.size());
    expand_cand_ids.reserve(mapping_ids.size());
    cand_embedding_ids.reserve(mapping_ids.size());

    for (size_t i = 0; i < mapping_ids.size(); ++i)
    {
        size_t unique_idx = mapping_ids[i];

        if (unique_idx >= unique_embeddings.size())
        {
            std::cerr << "[POST-PROCESS] Invalid mapping at i=" << i << ": unique_idx=" << unique_idx
                      << " >= unique_embeddings.size()=" << unique_embeddings.size() << std::endl;
            throw std::runtime_error("Invalid mapping index in expansion");
        }

        cand_seqs.push_back(unique_seqs[unique_idx]);
        expand_cand_ids.push_back(dense_ids[unique_idx]);
        cand_embedding_ids.push_back(unique_idx);
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Expansion completed in " << duration.count() << " ms" << std::endl;

    // Step 6: Batch rerank
    start_time = std::chrono::high_resolution_clock::now();
    std::cout << "[POST-PROCESS] Running batched reranker with pre-computed embeddings." << std::endl;
    auto results = batch_reranker(
        cand_seqs,
        expand_cand_ids,
        cand_embedding_ids,
        unique_embeddings,
        query_start_indices,
        query_embeddings,
        k_clusters);

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Reranking completed in " << duration.count() << " ms" << std::endl;

    // Step 7: Flatten results
    std::cout << "[POST-PROCESS] Formatting final results." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::string> final_seqs;
    std::vector<float> final_scores;
    std::vector<size_t> final_ids;
    final_seqs.reserve(results.size() * k);
    final_scores.reserve(results.size() * k);
    final_ids.reserve(results.size() * k);

    for (const auto &[seqs, scores, ids] : results)
    {
        final_seqs.insert(final_seqs.end(), seqs.begin(), seqs.end());
        final_scores.insert(final_scores.end(), scores.begin(), scores.end());
        final_ids.insert(final_ids.end(), ids.begin(), ids.end());
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Result formatting completed in " << duration.count() << " ms" << std::endl;

    return {final_seqs, final_scores, final_ids};
}

// Explicit template instantiations for different neighbor types
// Reformat output for different neighbor types
template std::pair<std::vector<size_t>, std::vector<float>> reformat_output<size_t>(const std::vector<std::vector<size_t>> &, const std::vector<std::vector<float>> &, size_t);
template std::pair<std::vector<size_t>, std::vector<float>> reformat_output<long int>(const std::vector<std::vector<long int>> &, const std::vector<std::vector<float>> &, size_t);

// Post-processing with Smith-Waterman reranker & dynamic lookup
template std::tuple<std::vector<std::string>, std::vector<int>, std::vector<size_t>> post_process_sw_dynamic<size_t>(const std::vector<std::vector<size_t>> &, const std::vector<std::vector<float>> &, const std::string &, const std::vector<std::string> &, size_t, size_t, size_t, size_t);
template std::tuple<std::vector<std::string>, std::vector<int>, std::vector<size_t>> post_process_sw_dynamic<long int>(const std::vector<std::vector<long int>> &, const std::vector<std::vector<float>> &, const std::string &, const std::vector<std::string> &, size_t, size_t, size_t, size_t);

// Post-processing with Smith-Waterman reranker & static lookup
template std::tuple<std::vector<std::string>, std::vector<int>, std::vector<size_t>> post_process_sw_static<size_t>(const std::vector<std::vector<size_t>> &, const std::vector<std::vector<float>> &, const std::vector<std::string> &, const std::vector<std::string> &, size_t, size_t, size_t, size_t);
template std::tuple<std::vector<std::string>, std::vector<int>, std::vector<size_t>> post_process_sw_static<long int>(const std::vector<std::vector<long int>> &, const std::vector<std::vector<float>> &, const std::vector<std::string> &, const std::vector<std::string> &, size_t, size_t, size_t, size_t);

// Post-processing with L2 reranker & dynamic lookup
template std::tuple<std::vector<std::string>, std::vector<float>, std::vector<size_t>> post_process_l2_dynamic<size_t>(const std::vector<std::vector<size_t>> &, const std::vector<std::vector<float>> &, const std::string &, const std::vector<std::string> &, size_t, size_t, size_t, const std::vector<std::vector<float>> &, Vectorizer &, size_t);
template std::tuple<std::vector<std::string>, std::vector<float>, std::vector<size_t>> post_process_l2_dynamic<long int>(const std::vector<std::vector<long int>> &, const std::vector<std::vector<float>> &, const std::string &, const std::vector<std::string> &, size_t, size_t, size_t, const std::vector<std::vector<float>> &, Vectorizer &, size_t);

// Post-processing with L2 reranker & dynamic lookup with streaming output
template void post_process_l2_dynamic_streaming<size_t>(const std::vector<std::vector<size_t>> &, const std::vector<std::vector<float>> &, const std::string &, const std::vector<std::string> &, const std::vector<std::string> &, size_t, size_t, size_t, const std::vector<std::vector<float>> &, Vectorizer &, size_t, const std::string &, const std::string &);
template void post_process_l2_dynamic_streaming<long int>(const std::vector<std::vector<long int>> &, const std::vector<std::vector<float>> &, const std::string &, const std::vector<std::string> &, const std::vector<std::string> &, size_t, size_t, size_t, const std::vector<std::vector<float>> &, Vectorizer &, size_t, const std::string &, const std::string &);

// Post-processing with L2 reranker & static lookup
template std::tuple<std::vector<std::string>, std::vector<float>, std::vector<size_t>> post_process_l2_static<size_t>(const std::vector<std::vector<size_t>> &, const std::vector<std::vector<float>> &, const std::vector<std::string> &, const std::vector<std::string> &, size_t, size_t, size_t, const std::vector<std::vector<float>> &, Vectorizer &, size_t);
template std::tuple<std::vector<std::string>, std::vector<float>, std::vector<size_t>> post_process_l2_static<long int>(const std::vector<std::vector<long int>> &, const std::vector<std::vector<float>> &, const std::vector<std::string> &, const std::vector<std::string> &, size_t, size_t, size_t, const std::vector<std::vector<float>> &, Vectorizer &, size_t);