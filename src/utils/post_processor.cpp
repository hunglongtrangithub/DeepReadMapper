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

std::vector<std::string> find_sequences(const std::string &ref_genome, const std::vector<size_t> &ids, size_t ref_len, size_t stride)
{
    if (ids.empty())
    {
        return {};
    }

    // Sort and deduplicate ids to minimize redundant fetches
    std::vector<size_t> sorted_ids = ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    sorted_ids.erase(std::unique(sorted_ids.begin(), sorted_ids.end()), sorted_ids.end());

    std::vector<std::pair<size_t, size_t>> ranges;

    for (size_t i = 0; i < sorted_ids.size(); ++i)
    {
        size_t base_id = sorted_ids[i];
        size_t actual_position = base_id * stride;

        if (actual_position >= ref_genome.size())
        {
            continue;
        }

        size_t start = (actual_position >= stride - 1) ? actual_position - stride + 1 : 0;
        size_t end = std::min(actual_position + stride, ref_genome.size());

        if (start >= end)
        {
            continue;
        }

        if (!ranges.empty() && start <= ranges.back().second)
        {
            ranges.back().second = std::max(ranges.back().second, end);
        }
        else
        {
            ranges.emplace_back(start, end);
        }
    }

    // Collect all unique positions from merged ranges
    std::vector<size_t> unique_positions;
    for (const auto &[start, end] : ranges)
    {
        for (size_t pos = start; pos < end; ++pos)
        {
            unique_positions.push_back(pos);
        }
    }

    // Start fetching sequences
    std::vector<std::string> results(unique_positions.size());

#pragma omp parallel for schedule(dynamic) num_threads(Config::PostProcess::NUM_THREADS)
    for (size_t i = 0; i < unique_positions.size(); ++i)
    {
        size_t pos = unique_positions[i];
        results[i] = find_sequence(ref_genome, pos, ref_len);
    }

    // Remove any empty strings (failed lookups) if needed
    results.erase(std::remove(results.begin(), results.end(), ""), results.end());

    return results;
}

// Wrapper for static lookup
std::vector<std::string> find_sequences(const std::vector<std::string> &ref_seqs, const std::vector<size_t> &ids, size_t stride)
{
    if (ids.empty())
    {
        return {};
    }

    // Sort and deduplicate ids to minimize redundant fetches
    std::vector<size_t> sorted_ids = ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    sorted_ids.erase(std::unique(sorted_ids.begin(), sorted_ids.end()), sorted_ids.end());

    std::vector<std::pair<size_t, size_t>> ranges;

    for (size_t i = 0; i < sorted_ids.size(); ++i)
    {
        size_t base_id = sorted_ids[i];
        size_t actual_position = base_id * stride;

        if (actual_position >= ref_seqs.size())
        {
            // throw std::out_of_range("Find out bound ID in find_sequences: " + std::to_string(actual_position) + " >= " + std::to_string(ref_seqs.size()));
            continue;
        }

        size_t start = (actual_position >= stride - 1) ? actual_position - stride + 1 : 0;
        size_t end = std::min(actual_position + stride, ref_seqs.size());

        if (start >= end)
        {
            continue;
        }

        if (!ranges.empty() && start <= ranges.back().second)
        {
            ranges.back().second = std::max(ranges.back().second, end);
        }
        else
        {
            ranges.emplace_back(start, end);
        }
    }

    // Collect all unique positions from merged ranges
    std::vector<size_t> unique_positions;
    for (const auto &[start, end] : ranges)
    {
        for (size_t pos = start; pos < end; ++pos)
        {
            unique_positions.push_back(pos);
        }
    }

    // Start fetching sequences
    std::vector<std::string> results(unique_positions.size());

    //* Use single-threaded for static lookup as each lookup is O(1)
    // #pragma omp parallel for schedule(dynamic) num_threads(Config::PostProcess::NUM_THREADS)
    for (size_t i = 0; i < unique_positions.size(); ++i)
    {
        size_t pos = unique_positions[i];
        // Add bounds checking in the lookup too
        if (pos < ref_seqs.size())
        {
            results[i] = find_sequence_static(ref_seqs, pos);
        }
        else
        {
            results[i] = ""; // Handle out of bounds
        }
    }

    // Remove any empty strings (failed lookups) if needed
    results.erase(std::remove(results.begin(), results.end(), ""), results.end());

    return results;
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

// Smith-Waterman version (templated)
template <typename NeighborType>
std::pair<std::vector<std::string>, std::vector<int>> post_process_sw(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::string &ref_genome,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k, size_t rerank_lim)
{
    std::cout << "[POST-PROCESS] Configs:" << std::endl;
    std::cout << "[POST-PROCESS] Metrics: sw_score" << std::endl;
    std::cout << "[POST-PROCESS] Neighbors type: " << typeid(NeighborType).name() << std::endl;

    auto converted_neighbors = convert_neighbors(neighbors);
    size_t total_queries = query_seqs.size();

    // Pre-allocate results for each query
    std::vector<std::vector<std::string>> all_final_seqs(total_queries);
    std::vector<std::vector<int>> all_scores(total_queries);

    // Thread-safe progress tracking
    std::atomic<size_t> completed_queries(0);

    // Hide cursor and create progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"post-processing"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    if (k < rerank_lim * stride)
    {
        throw std::runtime_error("Not enough candidates for dense translator. Ensure rerank_lim <= k / stride.");
    }
    // Parallelize across queries
    // #pragma omp parallel for num_threads(Config::PostProcess::NUM_THREADS) schedule(dynamic)

    for (size_t i = 0; i < total_queries; ++i)
    {
        // Select only first 5 candidates by HNSW score as preliminary ranking
        size_t num_cands_per_query = std::min(rerank_lim, converted_neighbors[i].size());

        std::vector<size_t> query_neighbors(converted_neighbors[i].begin(), converted_neighbors[i].begin() + num_cands_per_query);

        // Get candidate sequences
        std::vector<std::string> query_cand_seqs = find_sequences(ref_genome, query_neighbors, ref_len, stride);

        // Rerank candidates using Smith-Waterman
        auto [top_seqs, top_scores] = reranker(query_cand_seqs, query_seqs[i], k);

        // Store results for this query
        all_final_seqs[i] = std::move(top_seqs);
        all_scores[i] = std::move(top_scores);

        // Update progress bar (thread-safe)
        size_t current_completed = completed_queries.fetch_add(1) + 1;

        if (current_completed % 100 == 0)
        {
            size_t current_progress_percent = (current_completed * 100) / total_queries;
            progressBar.set_progress(current_progress_percent);
        }
    }

    // Flatten results
    std::vector<std::string> final_seqs;
    std::vector<int> scores;
    final_seqs.reserve(total_queries * k);
    scores.reserve(total_queries * k);

    for (size_t i = 0; i < total_queries; ++i)
    {
        final_seqs.insert(final_seqs.end(), all_final_seqs[i].begin(), all_final_seqs[i].end());
        scores.insert(scores.end(), all_scores[i].begin(), all_scores[i].end());
    }

    // Complete progress bar and show cursor
    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    return {final_seqs, scores};
}

// L2 distance version with dynamic lookup (templated)
template <typename NeighborType>
std::pair<std::vector<std::string>, std::vector<float>> post_process_l2_dynamic(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::string &ref_genome,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k,
    const std::vector<std::vector<float>> &query_embeddings,
    Vectorizer &vectorizer, size_t rerank_lim)
{
    std::cout << "[POST-PROCESS] Using dynamic lookup & batched reranker" << std::endl;
    std::cout << "[POST-PROCESS] Configs:" << std::endl;
    std::cout << "[POST-PROCESS] Metrics: L2 distance" << std::endl;
    std::cout << "[POST-PROCESS] Neighbors type: " << typeid(NeighborType).name() << std::endl;

    if (k < rerank_lim * stride)
    {
        throw std::runtime_error("Not enough candidates for dense translator. Ensure rerank_lim <= k / stride.");
    }

    auto converted_neighbors = convert_neighbors(neighbors);
    size_t total_queries = query_embeddings.size();

    // Step 1: Flatten all neighbor indices into a contiguous array and build mapping to track original query-cand pairs
    std::cout << "[POST-PROCESS] Flattening candidates for " << total_queries << " queries." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<size_t> all_neighbor_indices;
    std::vector<size_t> query_start_indices;

    for (size_t i = 0; i < total_queries; ++i)
    {
        query_start_indices.push_back(all_neighbor_indices.size());

        size_t num_cands_per_query = std::min(rerank_lim, converted_neighbors[i].size());

        all_neighbor_indices.insert(all_neighbor_indices.end(),
                                    converted_neighbors[i].begin(),
                                    converted_neighbors[i].begin() + num_cands_per_query);
    }
    query_start_indices.push_back(all_neighbor_indices.size()); // End marker

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Flattening completed in " << duration.count() << " ms" << std::endl;

    // Step 2: Single call to find_sequences for ALL candidates
    std::cout << "[POST-PROCESS] Fetching all candidate sequences." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::string> all_cand_seqs = find_sequences(ref_genome, all_neighbor_indices, ref_len, stride);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Fetching completed in " << duration.count() << " ms" << std::endl;

    // Early termination for stride == 1 (dense index)
    if (stride == 1)
    {
        std::cout << "[POST-PROCESS] Identified dense index (stride == 1), skipping reranker" << std::endl;

        std::vector<std::string> final_seqs;
        std::vector<float> final_scores;
        final_seqs.reserve(total_queries * k);
        final_scores.reserve(total_queries * k);

        size_t seq_idx = 0;
        for (size_t i = 0; i < total_queries; ++i)
        {
            size_t start_idx = query_start_indices[i];
            size_t end_idx = query_start_indices[i + 1];
            size_t actual_cands = end_idx - start_idx;

            // Add available sequences with their corresponding HNSW L2 distances
            for (size_t j = 0; j < std::min(k, actual_cands) && seq_idx < all_cand_seqs.size(); ++j, ++seq_idx)
            {
                final_seqs.push_back(all_cand_seqs[seq_idx]);

                if (j >= distances[i].size())
                {
                    throw std::runtime_error("Mismatch in distances size for query " + std::to_string(i));
                }

                final_scores.push_back(distances[i][j]);
            }
        }

        return {final_seqs, final_scores};
    }

    // Step 3: Pass flattened data to a modified batch_reranker
    std::cout << "[POST-PROCESS] Running batched reranker for all queries." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    auto batch_results = batch_reranker(all_cand_seqs, query_start_indices, query_embeddings, k, vectorizer);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Reranking completed in " << duration.count() << " ms" << std::endl;

    // Step 4: Flatten results in 2 vectors sequences and scores
    std::cout << "[POST-PROCESS] Format final results into 2 flat arrays (sequences and scores)." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::string> final_seqs;
    std::vector<float> final_scores;
    final_seqs.reserve(total_queries * k);
    final_scores.reserve(total_queries * k);

    for (const auto &[seqs, scores] : batch_results)
    {
        final_seqs.insert(final_seqs.end(), seqs.begin(), seqs.end());
        final_scores.insert(final_scores.end(), scores.begin(), scores.end());
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Result formatting completed in " << duration.count() << " ms" << std::endl;

    return {final_seqs, final_scores};
}

// L2 distance version with static lookup (templated)
template <typename NeighborType>
std::pair<std::vector<std::string>, std::vector<float>> post_process_l2_static(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::vector<std::string> &ref_seqs,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k,
    const std::vector<std::vector<float>> &query_embeddings,
    Vectorizer &vectorizer, size_t rerank_lim)
{
    std::cout << "[POST-PROCESS] Using static lookup & batched reranker" << std::endl;
    std::cout << "[POST-PROCESS] Configs:" << std::endl;
    std::cout << "[POST-PROCESS] Metrics: L2 distance" << std::endl;
    std::cout << "[POST-PROCESS] Neighbors type: " << typeid(NeighborType).name() << std::endl;

    if (k < rerank_lim * stride)
    {
        throw std::runtime_error("Not enough candidates for dense translator. Ensure rerank_lim <= k / stride.");
    }

    auto converted_neighbors = convert_neighbors(neighbors);
    size_t total_queries = query_embeddings.size();

    // Step 1: Flatten all neighbor indices into a contiguous array and build mapping to track original query-cand pairs
    std::cout << "[POST-PROCESS] Flattening candidates for " << total_queries << " queries." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<size_t> all_neighbor_indices;
    std::vector<size_t> query_start_indices;

    // Take top candidates per query to limit workload
    for (size_t i = 0; i < total_queries; ++i)
    {
        query_start_indices.push_back(all_neighbor_indices.size());

        size_t num_cands_per_query = std::min(rerank_lim, converted_neighbors[i].size());

        all_neighbor_indices.insert(all_neighbor_indices.end(),
                                    converted_neighbors[i].begin(),
                                    converted_neighbors[i].begin() + num_cands_per_query);
    }
    query_start_indices.push_back(all_neighbor_indices.size()); // End marker

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Flattening completed in " << duration.count() << " ms" << std::endl;

    // Step 2: Single call to find_sequences for ALL candidates
    std::cout << "[POST-PROCESS] Fetching all candidate sequences." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::string> all_cand_seqs = find_sequences(ref_seqs, all_neighbor_indices, stride);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Fetching completed in " << duration.count() << " ms" << std::endl;

    // Early termination for stride == 1 (dense index)
    if (stride == 1)
    {
        std::cout << "[POST-PROCESS] stride == 1 (dense), skipping reranker" << std::endl;

        std::vector<std::string> final_seqs;
        std::vector<float> final_scores;
        final_seqs.reserve(total_queries * k);
        final_scores.reserve(total_queries * k);

        size_t seq_idx = 0;
        for (size_t i = 0; i < total_queries; ++i)
        {
            size_t start_idx = query_start_indices[i];
            size_t end_idx = query_start_indices[i + 1];
            size_t actual_cands = end_idx - start_idx;

            // Add available sequences with their corresponding HNSW L2 distances
            for (size_t j = 0; j < std::min(k, actual_cands) && seq_idx < all_cand_seqs.size(); ++j, ++seq_idx)
            {
                final_seqs.push_back(all_cand_seqs[seq_idx]);

                if (j >= distances[i].size())
                {
                    throw std::runtime_error("Mismatch in distances size for query " + std::to_string(i));
                }

                final_scores.push_back(distances[i][j]);
            }
        }

        return {final_seqs, final_scores};
    }

    // Step 3: Pass flattened data to a batch_reranker (for stride > 1)
    std::cout << "[POST-PROCESS] Running batched reranker for all queries." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    auto batch_results = batch_reranker(all_cand_seqs, query_start_indices, query_embeddings, k, vectorizer);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Reranking completed in " << duration.count() << " ms" << std::endl;

    // Step 4: Flatten results in 2 vectors sequences and scores
    std::cout << "[POST-PROCESS] Format final results into 2 flat arrays (sequences and scores)." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::string> final_seqs;
    std::vector<float> final_scores;
    final_seqs.reserve(total_queries * k);
    final_scores.reserve(total_queries * k);

    for (const auto &[seqs, scores] : batch_results)
    {
        final_seqs.insert(final_seqs.end(), seqs.begin(), seqs.end());
        final_scores.insert(final_scores.end(), scores.begin(), scores.end());
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Result formatting completed in " << duration.count() << " ms" << std::endl;

    return {final_seqs, final_scores};
}

// Explicit template instantiations for different neighbor types
template std::pair<std::vector<size_t>, std::vector<float>> reformat_output<size_t>(const std::vector<std::vector<size_t>> &, const std::vector<std::vector<float>> &, size_t);
template std::pair<std::vector<size_t>, std::vector<float>> reformat_output<long int>(const std::vector<std::vector<long int>> &, const std::vector<std::vector<float>> &, size_t);

template std::pair<std::vector<std::string>, std::vector<int>> post_process_sw<size_t>(const std::vector<std::vector<size_t>> &, const std::vector<std::vector<float>> &, const std::string &, const std::vector<std::string> &, size_t, size_t, size_t, size_t);
template std::pair<std::vector<std::string>, std::vector<int>> post_process_sw<long int>(const std::vector<std::vector<long int>> &, const std::vector<std::vector<float>> &, const std::string &, const std::vector<std::string> &, size_t, size_t, size_t, size_t);

template std::pair<std::vector<std::string>, std::vector<float>> post_process_l2_dynamic<size_t>(const std::vector<std::vector<size_t>> &, const std::vector<std::vector<float>> &, const std::string &, const std::vector<std::string> &, size_t, size_t, size_t, const std::vector<std::vector<float>> &, Vectorizer &, size_t);
template std::pair<std::vector<std::string>, std::vector<float>> post_process_l2_dynamic<long int>(const std::vector<std::vector<long int>> &, const std::vector<std::vector<float>> &, const std::string &, const std::vector<std::string> &, size_t, size_t, size_t, const std::vector<std::vector<float>> &, Vectorizer &, size_t);

template std::pair<std::vector<std::string>, std::vector<float>> post_process_l2_static<size_t>(const std::vector<std::vector<size_t>> &, const std::vector<std::vector<float>> &, const std::vector<std::string> &, const std::vector<std::string> &, size_t, size_t, size_t, const std::vector<std::vector<float>> &, Vectorizer &, size_t);
template std::pair<std::vector<std::string>, std::vector<float>> post_process_l2_static<long int>(const std::vector<std::vector<long int>> &, const std::vector<std::vector<float>> &, const std::vector<std::string> &, const std::vector<std::string> &, size_t, size_t, size_t, const std::vector<std::vector<float>> &, Vectorizer &, size_t);