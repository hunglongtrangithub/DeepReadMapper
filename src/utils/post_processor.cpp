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

// Dynamic lookup version - returns both sequences and their dense IDs
std::pair<std::vector<std::string>, std::vector<size_t>> find_sequences(
    const std::string &ref_genome,
    const std::vector<size_t> &sparse_ids,
    size_t ref_len,
    size_t stride)
{
    if (sparse_ids.empty())
    {
        return {{}, {}};
    }

    // For dense index (stride==1), do direct lookup
    if (stride == 1)
    {
        std::vector<std::string> results;
        std::vector<size_t> expanded_ids;
        results.reserve(sparse_ids.size());
        expanded_ids.reserve(sparse_ids.size());

        for (size_t id : sparse_ids)
        {
            results.push_back(find_sequence(ref_genome, id, ref_len));
            expanded_ids.push_back(id); // ID stays same for stride=1
        }

        return {results, expanded_ids};
    }

    // For sparse index (stride>1), expand to neighboring ids
    std::vector<std::string> results;
    std::vector<size_t> expanded_ids;
    size_t expected_size = sparse_ids.size() * (2 * stride - 1);
    results.reserve(expected_size);
    expanded_ids.reserve(expected_size);

    for (size_t sparse_id : sparse_ids)
    {
        size_t actual_position = sparse_id * stride;

        if (actual_position >= ref_genome.size())
        {
            continue; // Skip out-of-bounds
        }

        size_t start = (actual_position >= stride - 1) ? actual_position - stride + 1 : 0;
        size_t end = std::min(actual_position + stride, ref_genome.size());

        // Add all sequences in the range WITH their dense IDs
        for (size_t pos = start; pos < end; ++pos)
        {
            results.push_back(find_sequence(ref_genome, pos, ref_len));
            expanded_ids.push_back(pos); // Store the actual dense position
        }
    }

    return {results, expanded_ids};
}

// Wrapper to return both sequences and their dense IDs
std::pair<std::vector<std::string>, std::vector<size_t>> find_sequences(
    const std::vector<std::string> &ref_seqs,
    const std::vector<size_t> &sparse_ids,
    size_t stride)
{
    if (sparse_ids.empty())
    {
        return {{}, {}};
    }

    if (stride == 1)
    {
        std::vector<std::string> results;
        std::vector<size_t> expanded_ids;
        results.reserve(sparse_ids.size());
        expanded_ids.reserve(sparse_ids.size());

        for (size_t id : sparse_ids)
        {
            if (id < ref_seqs.size())
            {
                results.push_back(ref_seqs[id]);
                expanded_ids.push_back(id); // ID stays same for stride=1
            }
        }
        return {results, expanded_ids};
    }

    // For sparse index (stride>1), expand to neighboring ids
    std::vector<std::string> results;
    std::vector<size_t> expanded_ids; // Track which dense ID each sequence corresponds to
    size_t expected_size = sparse_ids.size() * (2 * stride - 1);
    results.reserve(expected_size);
    expanded_ids.reserve(expected_size);

    for (size_t sparse_id : sparse_ids)
    {
        size_t actual_position = sparse_id * stride;

        if (actual_position >= ref_seqs.size())
        {
            continue; // Skip out-of-bounds
        }

        size_t start = (actual_position >= stride - 1) ? actual_position - stride + 1 : 0;
        size_t end = std::min(actual_position + stride, ref_seqs.size());

        // Add all sequences in the range WITH their dense IDs
        for (size_t pos = start; pos < end; ++pos)
        {
            if (pos < ref_seqs.size())
            {
                results.push_back(ref_seqs[pos]);
                expanded_ids.push_back(pos); // Store the actual dense position
            }
        }
    }

    return {results, expanded_ids};
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
    std::vector<std::vector<std::string>> all_final_seqs(total_queries);
    std::vector<std::vector<int>> all_scores(total_queries);
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

        // Get candidate sequences
        auto [query_cand_seqs, expanded_ids] = find_sequences(ref_genome, query_neighbors, ref_len, stride);

        // Rerank candidates using Smith-Waterman
        auto [top_seqs, top_scores, top_ids] = sw_reranker(query_cand_seqs, expanded_ids, query_seqs[i], k);

        // Store results for this query
        all_final_seqs[i] = std::move(top_seqs);
        all_scores[i] = std::move(top_scores);
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
        final_seqs.insert(final_seqs.end(), all_final_seqs[i].begin(), all_final_seqs[i].end());
        scores.insert(scores.end(), all_scores[i].begin(), all_scores[i].end());
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
    std::vector<std::vector<std::string>> all_final_seqs(total_queries);
    std::vector<std::vector<int>> all_scores(total_queries);
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

        // Get candidate sequences using static lookup
        auto [query_cand_seqs, expanded_ids] = find_sequences(ref_seqs, query_neighbors, stride);

        // Rerank candidates using Smith-Waterman
        auto [top_seqs, top_scores, top_ids] = sw_reranker(query_cand_seqs, expanded_ids, query_seqs[i], k);

        // Store results for this query
        all_final_seqs[i] = std::move(top_seqs);
        all_scores[i] = std::move(top_scores);
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
        final_seqs.insert(final_seqs.end(), all_final_seqs[i].begin(), all_final_seqs[i].end());
        scores.insert(scores.end(), all_scores[i].begin(), all_scores[i].end());
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

    if (k > k_clusters * 2 * stride)
    {
        throw std::runtime_error("Final k too large. Ensure k < k_clusters * 2 * stride to have enough candidates.");
    }

    auto converted_neighbors = convert_neighbors(neighbors);
    size_t total_queries = query_embeddings.size();

    // Pre-allocate final results
    std::vector<std::string> final_seqs;
    std::vector<float> final_scores;
    std::vector<size_t> final_ids;
    final_seqs.reserve(total_queries * k);
    final_scores.reserve(total_queries * k);
    final_ids.reserve(total_queries * k);

    // Configuration: Process queries in batches to avoid memory overflow
    const size_t QUERY_BATCH_SIZE = Config::Postprocess::QUERY_BATCH_SIZE;
    const size_t num_batches = (total_queries + QUERY_BATCH_SIZE - 1) / QUERY_BATCH_SIZE;

    std::cout << "[POST-PROCESS] Processing " << total_queries << " queries in " << num_batches << " batches (size :" << QUERY_BATCH_SIZE << ")" << std::endl;

    // Progress bar for batches
    indicators::show_console_cursor(false);
    indicators::ProgressBar batchProgressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"processing query batches"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx)
    {
        size_t batch_start = batch_idx * QUERY_BATCH_SIZE;
        size_t batch_end = std::min(batch_start + QUERY_BATCH_SIZE, total_queries);
        size_t batch_size = batch_end - batch_start;

        auto batch_start_time = std::chrono::high_resolution_clock::now();

        // Step 1: Flatten sparse IDs for THIS BATCH
        std::vector<size_t> sparse_ids;
        std::vector<size_t> query_start_indices;
        query_start_indices.reserve(batch_size + 1);

        for (size_t i = batch_start; i < batch_end; ++i)
        {
            query_start_indices.push_back(sparse_ids.size());
            size_t num_cands_per_query = std::min(k_clusters, converted_neighbors[i].size());
            sparse_ids.insert(sparse_ids.end(),
                              converted_neighbors[i].begin(),
                              converted_neighbors[i].begin() + num_cands_per_query);
        }
        query_start_indices.push_back(sparse_ids.size()); // End marker

        // Step 2: Fetch sequences for THIS BATCH
        auto [all_cand_seqs, dense_ids] = find_sequences(ref_genome, sparse_ids, ref_len, stride);

        // Step 2.5: Update query_start_indices after expansion
        std::vector<size_t> expanded_query_start_indices;
        expanded_query_start_indices.reserve(batch_size + 1);
        expanded_query_start_indices.push_back(0);
        
        size_t dense_offset = 0;
        for (size_t q = 0; q < batch_size; ++q)
        {
            size_t sparse_start = query_start_indices[q];
            size_t sparse_end = query_start_indices[q + 1];
            
            for (size_t s = sparse_start; s < sparse_end; ++s)
            {
                size_t sparse_id = sparse_ids[s];
                size_t actual_position = sparse_id * stride;
                
                if (actual_position >= ref_genome.size())
                {
                    continue;
                }
                
                size_t start_pos = (actual_position >= stride - 1) ? actual_position - stride + 1 : 0;
                size_t end_pos = std::min(actual_position + stride, ref_genome.size());
                size_t expansion_count = end_pos - start_pos;
                
                dense_offset += expansion_count;
            }
            expanded_query_start_indices.push_back(dense_offset);
        }
        
        query_start_indices = expanded_query_start_indices;

        // Early termination for stride == 1 (dense index)
        if (stride == 1)
        {
            for (size_t i = 0; i < batch_size; ++i)
            {
                size_t global_idx = batch_start + i;
                size_t start_idx = query_start_indices[i];
                size_t end_idx = query_start_indices[i + 1];
                size_t actual_cands = std::min(k, end_idx - start_idx);

                for (size_t j = 0; j < actual_cands; ++j)
                {
                    final_seqs.push_back(all_cand_seqs[start_idx + j]);

                    if (j >= distances[global_idx].size())
                    {
                        throw std::runtime_error("Mismatch in distances size for query " + std::to_string(global_idx));
                    }

                    final_scores.push_back(distances[global_idx][j]);
                    final_ids.push_back(converted_neighbors[global_idx][j]);
                }
            }
        }
        else
        {
            // Step 3: Rerank THIS BATCH
            // Extract batch embeddings
            std::vector<std::vector<float>> batch_embeddings(
                query_embeddings.begin() + batch_start,
                query_embeddings.begin() + batch_end);

            auto batch_results = batch_reranker(all_cand_seqs, dense_ids, query_start_indices, batch_embeddings, k, vectorizer);

            // Step 4: Accumulate results from THIS BATCH
            for (const auto &[seqs, scores, ids] : batch_results)
            {
                final_seqs.insert(final_seqs.end(), seqs.begin(), seqs.end());
                final_scores.insert(final_scores.end(), scores.begin(), scores.end());
                final_ids.insert(final_ids.end(), ids.begin(), ids.end());
            }
        }

        // Free batch memory immediately
        all_cand_seqs.clear();
        all_cand_seqs.shrink_to_fit();
        dense_ids.clear();
        dense_ids.shrink_to_fit();

        auto batch_end_time = std::chrono::high_resolution_clock::now();
        auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end_time - batch_start_time);

        // Update batch progress bar
        batchProgressBar.set_progress(((batch_idx + 1) * 100) / num_batches);
    }

    batchProgressBar.set_progress(100);
    indicators::show_console_cursor(true);

    std::cout << "\n[POST-PROCESS] All batches completed!" << std::endl;
    std::cout << "[POST-PROCESS] Total final sequences: " << final_seqs.size() << std::endl;

    return {final_seqs, final_scores, final_ids};
}

// L2 distance version with dynamic lookup with streaming output (templated)
template <typename NeighborType>
void post_process_l2_dynamic_streaming(  // Changed return type to void
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

    // Configuration: Process queries in batches
    const size_t QUERY_BATCH_SIZE = Config::Postprocess::QUERY_BATCH_SIZE;
    const size_t num_batches = (total_queries + QUERY_BATCH_SIZE - 1) / QUERY_BATCH_SIZE;

    std::cout << "[POST-PROCESS] Processing " << total_queries << " queries in " << num_batches << " batches (size: " << QUERY_BATCH_SIZE << ")" << std::endl;
    std::cout << "[POST-PROCESS] Streaming output to: " << sam_file << std::endl;

    // Progress bar for batches
    indicators::show_console_cursor(false);
    indicators::ProgressBar batchProgressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"processing & streaming batches"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx)
    {
        size_t batch_start = batch_idx * QUERY_BATCH_SIZE;
        size_t batch_end = std::min(batch_start + QUERY_BATCH_SIZE, total_queries);
        size_t batch_size = batch_end - batch_start;

        auto batch_start_time = std::chrono::high_resolution_clock::now();

        // Step 1: Flatten sparse IDs for THIS BATCH
        std::vector<size_t> sparse_ids;
        std::vector<size_t> query_start_indices;
        query_start_indices.reserve(batch_size + 1);

        for (size_t i = batch_start; i < batch_end; ++i)
        {
            query_start_indices.push_back(sparse_ids.size());
            size_t num_cands_per_query = std::min(k_clusters, converted_neighbors[i].size());
            sparse_ids.insert(sparse_ids.end(),
                              converted_neighbors[i].begin(),
                              converted_neighbors[i].begin() + num_cands_per_query);
        }
        query_start_indices.push_back(sparse_ids.size());

        // Step 2: Fetch sequences for THIS BATCH
        auto [all_cand_seqs, dense_ids] = find_sequences(ref_genome, sparse_ids, ref_len, stride);

        // Step 2.5: Update query_start_indices after expansion
        std::vector<size_t> expanded_query_start_indices;
        expanded_query_start_indices.reserve(batch_size + 1);
        expanded_query_start_indices.push_back(0);
        
        size_t dense_offset = 0;
        for (size_t q = 0; q < batch_size; ++q)
        {
            size_t sparse_start = query_start_indices[q];
            size_t sparse_end = query_start_indices[q + 1];
            
            for (size_t s = sparse_start; s < sparse_end; ++s)
            {
                size_t sparse_id = sparse_ids[s];
                size_t actual_position = sparse_id * stride;
                
                if (actual_position >= ref_genome.size())
                    continue;
                
                size_t start_pos = (actual_position >= stride - 1) ? actual_position - stride + 1 : 0;
                size_t end_pos = std::min(actual_position + stride, ref_genome.size());
                dense_offset += (end_pos - start_pos);
            }
            expanded_query_start_indices.push_back(dense_offset);
        }
        
        query_start_indices = expanded_query_start_indices;

        // Batch-level results
        std::vector<std::string> batch_seqs;
        std::vector<float> batch_scores;
        std::vector<size_t> batch_ids;
        batch_seqs.reserve(batch_size * k);
        batch_scores.reserve(batch_size * k);
        batch_ids.reserve(batch_size * k);

        // Process based on stride
        if (stride == 1)
        {
            for (size_t i = 0; i < batch_size; ++i)
            {
                size_t global_idx = batch_start + i;
                size_t start_idx = query_start_indices[i];
                size_t end_idx = query_start_indices[i + 1];
                size_t actual_cands = std::min(k, end_idx - start_idx);

                for (size_t j = 0; j < actual_cands; ++j)
                {
                    batch_seqs.push_back(all_cand_seqs[start_idx + j]);
                    batch_scores.push_back(distances[global_idx][j]);
                    batch_ids.push_back(converted_neighbors[global_idx][j]);
                }
            }
        }
        else
        {
            // Step 3: Rerank THIS BATCH
            std::vector<std::vector<float>> batch_embeddings(
                query_embeddings.begin() + batch_start,
                query_embeddings.begin() + batch_end);

            auto batch_results = batch_reranker(all_cand_seqs, dense_ids, query_start_indices, batch_embeddings, k, vectorizer);

            // Step 4: Flatten batch results
            for (const auto &[seqs, scores, ids] : batch_results)
            {
                batch_seqs.insert(batch_seqs.end(), seqs.begin(), seqs.end());
                batch_scores.insert(batch_scores.end(), scores.begin(), scores.end());
                batch_ids.insert(batch_ids.end(), ids.begin(), ids.end());
            }
        }

        // STREAM TO DISK IMMEDIATELY
        bool write_header = (batch_idx == 0);
        write_sam_streaming(batch_seqs, batch_scores, query_seqs, query_ids, batch_ids,
                           ref_name, ref_len, k, sam_file, batch_start, write_header);

        // Free batch memory
        all_cand_seqs.clear();
        all_cand_seqs.shrink_to_fit();
        batch_seqs.clear();
        batch_seqs.shrink_to_fit();

        auto batch_end_time = std::chrono::high_resolution_clock::now();
        auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end_time - batch_start_time);

        batchProgressBar.set_progress(((batch_idx + 1) * 100) / num_batches);
    }

    batchProgressBar.set_progress(100);
    indicators::show_console_cursor(true);

    std::cout << "\n[POST-PROCESS] All batches completed and streamed to disk!" << std::endl;
}


// L2 distance version with static lookup (templated)
template <typename NeighborType>
std::tuple<std::vector<std::string>, std::vector<float>, std::vector<size_t>> post_process_l2_static(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::vector<std::string> &ref_seqs,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k,
    const std::vector<std::vector<float>> &query_embeddings,
    Vectorizer &vectorizer, size_t k_clusters)
{
    std::cout << "[POST-PROCESS] Using static lookup & batched reranker" << std::endl;
    std::cout << "[POST-PROCESS] Configs:" << std::endl;
    std::cout << "[POST-PROCESS] Metrics: L2 distance" << std::endl;
    std::cout << "[POST-PROCESS] Neighbors type: " << typeid(NeighborType).name() << std::endl;

    if (k > k_clusters * 2 * stride && stride > 1)
    {
        throw std::runtime_error("Final k too large. Ensure k < k_clusters * 2 * stride to have enough candidates.");
    }

    auto converted_neighbors = convert_neighbors(neighbors);
    size_t total_queries = query_embeddings.size();

    // Step 1: Flatten all neighbor indices into a contiguous array and build mapping to track original query-cand pairs
    std::cout << "[POST-PROCESS] Flattening candidates for " << total_queries << " queries." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<size_t> sparse_ids;
    std::vector<size_t> query_start_indices;

    // Take top candidates per query to limit workload
    for (size_t i = 0; i < total_queries; ++i)
    {
        query_start_indices.push_back(sparse_ids.size());

        // For dense index (stride==1), use k directly since no reranking needed
        // For sparse index (stride>1), use k_clusters to limit reranking workload
        size_t num_cands_per_query = (stride == 1)
                                         ? std::min(k, converted_neighbors[i].size())
                                         : std::min(k_clusters, converted_neighbors[i].size());

        sparse_ids.insert(sparse_ids.end(),
                          converted_neighbors[i].begin(),
                          converted_neighbors[i].begin() + num_cands_per_query);
    }
    query_start_indices.push_back(sparse_ids.size()); // End marker

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Flattening completed in " << duration.count() << " ms" << std::endl;

    // Step 2: Single call to find_sequences for ALL candidates
    std::cout << "[POST-PROCESS] Fetching all candidate sequences." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    auto [all_cand_seqs, dense_ids] = find_sequences(ref_seqs, sparse_ids, stride);

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Fetching completed in " << duration.count() << " ms" << std::endl;

    // Step 2.5: Update query_start_indices
    std::cout << "[POST-PROCESS] Adjusting query start indices after dynamic expansion." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<size_t> expanded_query_start_indices;
    expanded_query_start_indices.reserve(query_seqs.size() + 1);
    expanded_query_start_indices.push_back(0);
    
    size_t dense_offset = 0;
    for (size_t q = 0; q < query_seqs.size(); ++q)
    {
        size_t sparse_start = query_start_indices[q];
        size_t sparse_end = query_start_indices[q + 1];
        
        // Calculate how many dense candidates this query got
        for (size_t s = sparse_start; s < sparse_end; ++s)
        {
            size_t sparse_id = sparse_ids[s];
            size_t actual_position = sparse_id * stride;
            
            // Calculate expansion for this sparse ID (same logic as find_sequences)
            size_t start_pos = (actual_position >= stride - 1) ? actual_position - stride + 1 : 0;
            size_t end_pos = std::min(actual_position + stride, ref_seqs.size());
            size_t expansion_count = end_pos - start_pos;
            
            dense_offset += expansion_count;
        }
        expanded_query_start_indices.push_back(dense_offset);
    }
    
    // Replace old indices
    query_start_indices = expanded_query_start_indices;
    
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Adjustment completed in " << duration.count() << " ms" << std::endl;
        

    // Early termination for stride == 1 (dense index)
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
            size_t start_idx = query_start_indices[i];
            size_t end_idx = query_start_indices[i + 1];
            size_t actual_cands = std::min(k, end_idx - start_idx);

            // Use start_idx to index into all_cand_seqs properly
            for (size_t j = 0; j < actual_cands; ++j)
            {
                final_seqs.push_back(all_cand_seqs[start_idx + j]);

                if (j >= distances[i].size())
                {
                    throw std::runtime_error("Mismatch in distances size for query " + std::to_string(i));
                }

                final_scores.push_back(distances[i][j]);
                final_ids.push_back(converted_neighbors[i][j]);
            }
        }

        return {final_seqs, final_scores, final_ids};
    }

    // Step 3: Pass flattened data to a batch_reranker (for stride > 1)
    std::cout << "[POST-PROCESS] Running batched reranker for all queries." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    auto batch_results = batch_reranker(all_cand_seqs, dense_ids, query_start_indices, query_embeddings, k, vectorizer);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Reranking completed in " << duration.count() << " ms" << std::endl;

    // Step 4: Flatten results in 3 vectors
    std::cout << "[POST-PROCESS] Format final results into 3 flat arrays (sequences, scores, ids)." << std::endl;
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