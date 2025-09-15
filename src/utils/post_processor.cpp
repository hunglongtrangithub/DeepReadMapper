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

// Reformat hnsw output
std::pair<std::vector<size_t>, std::vector<float>> reformat_output(const std::vector<std::vector<size_t>> &neighbors, const std::vector<std::vector<float>> &distances, size_t k)
{
    size_t n_rows = neighbors.size();

    // Flatten the 2D vectors into 1D arrays
    std::vector<size_t> host_indices(n_rows * k);
    std::vector<float> host_distances(n_rows * k);

    for (size_t i = 0; i < n_rows; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            host_indices[i * k + j] = neighbors[i][j];
            host_distances[i * k + j] = distances[i][j];
        }
    }

    return {host_indices, host_distances};
}

// Overload for long int type
std::pair<std::vector<size_t>, std::vector<float>> reformat_output(const std::vector<std::vector<long int>> &neighbors, const std::vector<std::vector<float>> &distances, size_t k)
{
    // Convert long int to size_t
    std::vector<std::vector<size_t>> neighbors_size_t(neighbors.size());
    for (size_t i = 0; i < neighbors.size(); ++i)
    {
        neighbors_size_t[i].assign(neighbors[i].begin(), neighbors[i].end());
    }
    // Call the existing function
    return reformat_output(neighbors_size_t, distances, k);
}

std::string find_sequence(const std::vector<std::string> &ref_seqs, size_t id, size_t ref_len)
{
    if (id < 0 || id >= ref_seqs.size())
    {
        throw std::out_of_range("ID " + std::to_string(id) + " out of range for reference sequences of size " + std::to_string(ref_seqs.size()));
    }
    std::string cand_seq = ref_seqs[id];

    // Remove PREFIX and POSTFIX
    return cand_seq.substr(1, ref_len);
}

std::vector<std::string> find_sequences(const std::vector<std::string> &ref_seqs, const std::vector<size_t> &ids, size_t ref_len, size_t stride)
{
    std::vector<std::string> results;

    if (ids.empty()) {
        std::cout << "[DEBUG] find_sequences: Empty ids vector" << std::endl;
        return results;
    }

    // Debug initial state
    std::cout << "[DEBUG] find_sequences called with:" << std::endl;
    std::cout << "[DEBUG]   ref_seqs.size() = " << ref_seqs.size() << std::endl;
    std::cout << "[DEBUG]   ids.size() = " << ids.size() << std::endl;
    std::cout << "[DEBUG]   ref_len = " << ref_len << std::endl;
    std::cout << "[DEBUG]   stride = " << stride << std::endl;

    // Show first few IDs
    std::cout << "[DEBUG] First 10 sparse_ids: ";
    for (size_t i = 0; i < std::min(size_t(10), ids.size()); ++i) {
        std::cout << ids[i] << " ";
    }
    std::cout << std::endl;

    // Step 1: Sort candidates and merge overlapping ranges
    std::vector<size_t> sorted_ids = ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());

    std::vector<std::pair<size_t, size_t>> ranges; // [start, end) pairs

    size_t skipped_ranges = 0;
    size_t valid_ranges = 0;
    size_t out_of_bounds = 0;

    for (size_t i = 0; i < sorted_ids.size(); ++i)
    {
        size_t base_id = sorted_ids[i];
        
        // Debug first few iterations
        if (i < 5) {
            std::cout << "[DEBUG] Processing base_id[" << i << "] = " << base_id << std::endl;
        }

        // Normalize base_id, shift from 0, 1, 2 to 0, stride, 2*stride, ...
        size_t actual_position = base_id * stride;
        
        // Check if out of bounds
        if (actual_position >= ref_seqs.size()) {
            out_of_bounds++;
            if (i < 5) {
                std::cout << "[DEBUG]   actual_position " << actual_position << " >= ref_seqs.size() " << ref_seqs.size() << " (OUT OF BOUNDS)" << std::endl;
            }
            continue;
        }

        size_t start = (actual_position >= stride - 1) ? actual_position - stride + 1 : 0;
        size_t end = std::min(actual_position + stride, ref_seqs.size());

        // Debug range calculation
        if (i < 5) {
            std::cout << "[DEBUG]   actual_position = " << actual_position << std::endl;
            std::cout << "[DEBUG]   start = " << start << ", end = " << end << std::endl;
        }

        // Check if range is valid
        if (start >= end) {
            skipped_ranges++;
            if (i < 5) {
                std::cout << "[DEBUG]   INVALID RANGE: start >= end" << std::endl;
            }
            continue;
        }

        valid_ranges++;

        // Merge with previous range if overlapping
        if (!ranges.empty() && start <= ranges.back().second)
        {
            size_t old_end = ranges.back().second;
            ranges.back().second = std::max(ranges.back().second, end);
            if (i < 5) {
                std::cout << "[DEBUG]   MERGED with previous range: [" << ranges.back().first << ", " << old_end << ") -> [" << ranges.back().first << ", " << ranges.back().second << ")" << std::endl;
            }
        }
        else
        {
            ranges.emplace_back(start, end);
            if (i < 5) {
                std::cout << "[DEBUG]   NEW range: [" << start << ", " << end << ")" << std::endl;
            }
        }
    }

    std::cout << "[DEBUG] Range processing summary:" << std::endl;
    std::cout << "[DEBUG]   Valid ranges: " << valid_ranges << std::endl;
    std::cout << "[DEBUG]   Skipped ranges: " << skipped_ranges << std::endl;
    std::cout << "[DEBUG]   Out of bounds: " << out_of_bounds << std::endl;
    std::cout << "[DEBUG]   Final merged ranges: " << ranges.size() << std::endl;

    // Step 2: Collect all unique positions from merged ranges
    std::vector<size_t> unique_positions;
    size_t total_positions = 0;

    for (size_t i = 0; i < ranges.size(); ++i)
    {
        const auto &[start, end] = ranges[i];
        size_t range_size = end - start;
        total_positions += range_size;

        if (i < 3) { // Debug first few ranges
            std::cout << "[DEBUG] Range[" << i << "]: [" << start << ", " << end << ") size=" << range_size << std::endl;
        }

        for (size_t pos = start; pos < end; ++pos)
        {
            unique_positions.push_back(pos);
        }
    }

    std::cout << "[DEBUG] Total unique positions to fetch: " << total_positions << std::endl;
    std::cout << "[DEBUG] unique_positions.size(): " << unique_positions.size() << std::endl;

    // Step 3: Fetch sequences
    results.reserve(unique_positions.size());
    size_t fetch_errors = 0;

    for (size_t i = 0; i < unique_positions.size(); ++i)
    {
        size_t pos = unique_positions[i];
        
        try {
            std::string seq = find_sequence(ref_seqs, pos, ref_len);
            results.push_back(seq);
            
            // Debug first few fetches
            if (i < 3) {
                std::cout << "[DEBUG] Fetched seq[" << i << "] from pos " << pos << ": length=" << seq.length() << std::endl;
            }
        }
        catch (const std::exception& e) {
            fetch_errors++;
            if (fetch_errors <= 5) { // Only show first few errors
                std::cout << "[DEBUG] ERROR fetching pos " << pos << ": " << e.what() << std::endl;
            }
        }
    }

    std::cout << "[DEBUG] Fetch summary:" << std::endl;
    std::cout << "[DEBUG]   Successful fetches: " << results.size() << std::endl;
    std::cout << "[DEBUG]   Fetch errors: " << fetch_errors << std::endl;
    std::cout << "[DEBUG] Final results.size(): " << results.size() << std::endl;

    return results;
}
// Smith-Waterman version (size_t neighbors)
std::pair<std::vector<std::string>, std::vector<int>> post_process(
    const std::vector<std::vector<size_t>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::vector<std::string> &ref_seqs,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k)
{
    std::cout << "[POST-PROCESS] Configs:" << std::endl;
    std::cout << "[POST-PROCESS] Metrics: sw_score" << std::endl;
    std::cout << "[POST-PROCESS] Neighbors type: size_t" << std::endl;
    return post_process_core<std::string, int>(neighbors, distances, ref_seqs, query_seqs, ref_len, stride, k, [](const std::vector<std::string> &cand_seqs, const std::string &query_seq, size_t k)
                                               { return reranker(cand_seqs, query_seq, k); });
}

// Smith-Waterman version (long int neighbors)
std::pair<std::vector<std::string>, std::vector<int>> post_process(
    const std::vector<std::vector<long int>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::vector<std::string> &ref_seqs,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k)
{

    std::cout << "[POST-PROCESS] Configs:" << std::endl;
    std::cout << "[POST-PROCESS] Metrics: sw_score" << std::endl;
    std::cout << "[POST-PROCESS] Neighbors type: long int" << std::endl;

    // Convert long int to size_t
    std::vector<std::vector<size_t>> neighbors_size_t(neighbors.size());
    for (size_t i = 0; i < neighbors.size(); ++i)
    {
        neighbors_size_t[i].assign(neighbors[i].begin(), neighbors[i].end());
    }

    // Call the size_t version
    return post_process(neighbors_size_t, distances, ref_seqs, query_seqs, ref_len, stride, k);
}

// L2 distance version (size_t neighbors)
std::pair<std::vector<std::string>, std::vector<float>> post_process(
    const std::vector<std::vector<size_t>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::vector<std::string> &ref_seqs,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k,
    const std::vector<std::vector<float>> &query_embeddings,
    Vectorizer &vectorizer)
{
    std::cout << "[POST-PROCESS] Using batched reranker." << std::endl;
    std::cout << "[POST-PROCESS] Configs:" << std::endl;
    std::cout << "[POST-PROCESS] Metrics: L2 distance (batched)" << std::endl;
    std::cout << "[POST-PROCESS] Neighbors type: size_t" << std::endl;

    size_t total_queries = query_embeddings.size();

    // Step 1: Flatten all neighbor indices into a contiguous array and build mapping to track original quer-cand pairs
    std::cout << "[POST-PROCESS] Flattening candidates for " << total_queries << " queries." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<size_t> all_neighbor_indices;
    std::vector<size_t> query_start_indices;

    for (size_t i = 0; i < total_queries; ++i)
    {
        query_start_indices.push_back(all_neighbor_indices.size());

        size_t num_cands_per_query = std::min(size_t(5), neighbors[i].size());

        all_neighbor_indices.insert(all_neighbor_indices.end(),
                                    neighbors[i].begin(),
                                    neighbors[i].begin() + num_cands_per_query);
    }
    query_start_indices.push_back(all_neighbor_indices.size()); // End marker

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Flattening completed in " << duration.count() << " ms" << std::endl;

    std::cout << "[DEBUG] Total neighbor indices collected: " << all_neighbor_indices.size() << std::endl;

    // Step 2: Single call to find_sequences for ALL candidates
    std::cout << "[POST-PROCESS] Fetching all candidate sequences." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::string> all_cand_seqs = find_sequences(ref_seqs, all_neighbor_indices, ref_len, stride);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[POST-PROCESS] Fetching completed in " << duration.count() << " ms" << std::endl;

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

// L2 distance version (long int neighbors)
std::pair<std::vector<std::string>, std::vector<float>> post_process(
    const std::vector<std::vector<long int>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::vector<std::string> &ref_seqs,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k,
    const std::vector<std::vector<float>> &query_embeddings,
    Vectorizer &vectorizer)
{
    std::cout << "[POST-PROCESS] Configs:" << std::endl;
    std::cout << "[POST-PROCESS] Metrics: L2 distance" << std::endl;
    std::cout << "[POST-PROCESS] Neighbors type: long int" << std::endl;
    // Convert long int to size_t
    std::vector<std::vector<size_t>> neighbors_size_t(neighbors.size());
    for (size_t i = 0; i < neighbors.size(); ++i)
    {
        neighbors_size_t[i].assign(neighbors[i].begin(), neighbors[i].end());
    }

    // Call the size_t version
    return post_process(neighbors_size_t, distances, ref_seqs, query_seqs, ref_len, stride, k, query_embeddings, vectorizer);
}