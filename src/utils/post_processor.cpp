#include "post_processor.hpp"
#include "progressbar.h"

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
        results.resize(unique_positions.size());


        for (size_t pos : unique_positions)
        {
            results.push_back(find_sequence(ref_seqs, pos, ref_len));
        }
        
    }

    return results;
}

int calc_sw_score_avx2(const std::string &seq1, const std::string &seq2)
{
    std::cout << "Using AVX2 optimized Smith-Waterman" << std::endl;
    // TODO: Implement AVX2 optimized Smith-Waterman here
    // https://www.danysoft.com/estaticos/free/24%20-%20Case%20study%20-%20Pairwise%20sequence%20alignment%20with%20the%20Smith-Waterman%20algorithm.pdf
    return 0; // Placeholder
}

int calc_sw_score(const std::string &seq1, const std::string &seq2)
{
    // #ifdef __AVX2__
    //     // AVX2 optimized version here
    //     return calc_sw_score_avx2(seq1, seq2);
    // #else
    // Fallback scalar version
    // Scoring parameters to match Python implementation exactly
    const int match = 1;
    const int mismatch = -1;
    const int gap_penalty = -1; // gap penalty of 1 in Python = -1 score

    size_t len1 = seq1.size();
    size_t len2 = seq2.size();

    // Create DP matrix
    std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1, 0));
    int max_score = 0;

    // Fill DP matrix using standard Smith-Waterman with linear gap penalty
    for (size_t i = 1; i <= len1; ++i)
    {
        for (size_t j = 1; j <= len2; ++j)
        {
            int score_match = dp[i - 1][j - 1] + (seq1[i - 1] == seq2[j - 1] ? match : mismatch);
            int score_delete = dp[i - 1][j] + gap_penalty;
            int score_insert = dp[i][j - 1] + gap_penalty;

            dp[i][j] = std::max({0, score_match, score_delete, score_insert});
            max_score = std::max(max_score, dp[i][j]);
        }
    }

    return max_score;
    // #endif
}

float calc_l2_dist(const std::vector<float> &vec1, const std::vector<float> &vec2)
{
    if (vec1.size() != vec2.size())
    {
        throw std::invalid_argument("Vector sizes mismatch: " + std::to_string(vec1.size()) + " vs " + std::to_string(vec2.size()));
    }

    float sum = 0.0f;
    for (size_t i = 0; i < vec1.size(); ++i)
    {
        float diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::pair<std::vector<std::string>, std::vector<int>> sw_reranker(const std::vector<std::string> &cand_seqs, const std::string &query_seq, size_t k, size_t stride)
{
    size_t num_cands = cand_seqs.size();
    if (num_cands == 0 || k == 0)
        return {{}, {}};

    // Step 1: Compute SW scores for all candidates
    std::vector<int> scores(num_cands, 0);

    for (size_t i = 0; i < num_cands; ++i)
    {
        scores[i] = calc_sw_score(cand_seqs[i], query_seq);
    }

    // Step 2: Sort candidates by score
    std::vector<std::string> top_seqs;
    std::vector<int> top_scores;

    size_t actual_k = std::min(k, num_cands);
    top_seqs.reserve(actual_k);
    top_scores.reserve(actual_k);

    std::vector<size_t> indices(num_cands);
    std::iota(indices.begin(), indices.end(), 0);
    
    // FIX: Only partial_sort up to actual_k, not k
    std::partial_sort(indices.begin(), indices.begin() + actual_k, indices.end(), 
                      [&scores](size_t i1, size_t i2) { return scores[i1] > scores[i2]; });

    // Step 3: Collect top actual_k sequences and scores
    for (size_t i = 0; i < actual_k; ++i)
    {
        top_seqs.push_back(cand_seqs[indices[i]]);
        top_scores.push_back(scores[indices[i]]);
    }

    return {top_seqs, top_scores};
}

std::pair<std::vector<std::string>, std::vector<int>> post_process(const std::vector<std::vector<size_t>> &neighbors, const std::vector<std::vector<float>> &distances, const std::vector<std::string> &ref_seqs, const std::vector<std::string> &query_seqs, size_t ref_len, size_t stride, size_t k)
{
    //TODO: Implement L2 reranker by passed in Vectorizer and query embeddings, then find distances between embeddings
    size_t total_queries = query_seqs.size();
    
    // Pre-allocate results for each query
    std::vector<std::vector<std::string>> all_final_seqs(total_queries);
    std::vector<std::vector<int>> all_sw_scores(total_queries);
    
    // Thread-safe progress tracking
    std::atomic<size_t> completed_queries(0);
    
    // Hide cursor and create progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"post-processing"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    // Parallelize across queries
#pragma omp parallel for num_threads(Config::PostProcess::NUM_THREADS) schedule(dynamic)
    for (size_t i = 0; i < total_queries; ++i)
    {
        // Extract neighbors for this specific query
        // std::vector<size_t> query_neighbors(neighbors[i].begin(), neighbors[i].end());

        //* Debug, select only first 5 candidates (filter by HNSW score as preliminary ranking)
        std::vector<size_t> query_neighbors(neighbors[i].begin(), neighbors[i].begin() + std::min(size_t(5), neighbors[i].size()));

        // Get candidate sequences
        std::vector<std::string> query_cand_seqs = find_sequences(ref_seqs, query_neighbors, ref_len, stride);

        // Rerank candidates
        auto [top_seqs, top_scores] = sw_reranker(query_cand_seqs, query_seqs[i], static_cast<size_t>(k), stride);

        // Store results for this query
        all_final_seqs[i] = std::move(top_seqs);
        all_sw_scores[i] = std::move(top_scores);

        // Update progress bar (thread-safe)
        size_t current_completed = completed_queries.fetch_add(1) + 1;

        if (current_completed % 100 == 0) {
            size_t current_progress_percent = (current_completed * 100) / total_queries;
            progressBar.set_progress(current_progress_percent);
        }
    }

    // Flatten results
    std::vector<std::string> final_seqs;
    std::vector<int> sw_scores;
    final_seqs.reserve(total_queries * k);
    sw_scores.reserve(total_queries * k);
    
    for (size_t i = 0; i < total_queries; ++i)
    {
        final_seqs.insert(final_seqs.end(), all_final_seqs[i].begin(), all_final_seqs[i].end());
        sw_scores.insert(sw_scores.end(), all_sw_scores[i].begin(), all_sw_scores[i].end());
    }

    // Complete progress bar and show cursor
    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    return {final_seqs, sw_scores};
}

// Overload to accept long int type for neighbors
std::pair<std::vector<std::string>, std::vector<int>> post_process(const std::vector<std::vector<long int>> &neighbors, const std::vector<std::vector<float>> &distances, const std::vector<std::string> &ref_seqs, const std::vector<std::string> &query_seqs, size_t ref_len, size_t stride, size_t k)
{
    // Convert long int to size_t
    std::vector<std::vector<size_t>> neighbors_size_t(neighbors.size());
    for (size_t i = 0; i < neighbors.size(); ++i)
    {
        neighbors_size_t[i].assign(neighbors[i].begin(), neighbors[i].end());
    }
    
    // Call the existing function
    return post_process(neighbors_size_t, distances, ref_seqs, query_seqs, ref_len, stride, k);
}