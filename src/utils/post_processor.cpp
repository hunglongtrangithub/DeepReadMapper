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

    if (ids.empty())
        return results;

    // Step 1: Sort candidates and merge overlapping ranges
    std::vector<size_t> sorted_ids = ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());

    std::vector<std::pair<size_t, size_t>> ranges; // [start, end) pairs

    for (size_t base_id : sorted_ids)
    {
        // Normalize base_id, shift from 0, 1, 2 to 0, stride, 2*stride, ...
        base_id *= stride;
        size_t start = (base_id >= stride - 1) ? base_id - stride + 1 : 0;
        size_t end = std::min(base_id + stride, ref_seqs.size());

        // Merge with previous range if overlapping
        if (!ranges.empty() && start <= ranges.back().second)
        {
            ranges.back().second = std::max(ranges.back().second, end);
        }
        else
        {
            ranges.emplace_back(start, end);
        }
    }

    // Step 2: Collect all unique positions from merged ranges
    std::vector<size_t> unique_positions;
    for (const auto &[start, end] : ranges)
    {
        for (size_t pos = start; pos < end; ++pos)
        {
            unique_positions.push_back(pos);
        }
    }

    // Step 3: Fetch sequences
    results.reserve(unique_positions.size());
    for (size_t pos : unique_positions)
    {
        results.push_back(find_sequence(ref_seqs, pos, ref_len));
    }

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
    
    // Step 1: Collect all candidate sequences for all queries
    std::vector<std::vector<std::string>> all_cand_seqs(total_queries);
    for (size_t i = 0; i < total_queries; ++i) {
        std::vector<size_t> query_neighbors(neighbors[i].begin(),
                                           neighbors[i].begin() + std::min(size_t(5), neighbors[i].size()));
        all_cand_seqs[i] = find_sequences(ref_seqs, query_neighbors, ref_len, stride);
    }
    
    // Step 2: Batch rerank all candidates
    auto batch_results = batch_reranker(all_cand_seqs, query_embeddings, k, vectorizer);
    
    // Step 3: Flatten results
    std::vector<std::string> final_seqs;
    std::vector<float> final_scores;
    final_seqs.reserve(total_queries * k);
    final_scores.reserve(total_queries * k);
    
    for (const auto &[seqs, scores] : batch_results) {
        final_seqs.insert(final_seqs.end(), seqs.begin(), seqs.end());
        final_scores.insert(final_scores.end(), scores.begin(), scores.end());

    }
    
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