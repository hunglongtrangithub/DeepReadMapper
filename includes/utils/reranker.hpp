#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <utility>
#include <numeric>
#include "vectorize.hpp"
#include "metrics.hpp"

//TODO: Implement Banded Smith-Waterman for faster reranking
/// @brief A reranker that takes in EF * 2 * stride candidates and returns top k candidates as (seq, scores, ids) using Smith-Waterman as alignment metric
/// @param cand_seqs Vector of candidate sequences (removed PREFIX/POSTFIX)
/// @param cand_ids Vector of candidate IDs corresponding to cand_seqs
/// @param query_seq Query sequence to align against (removed PREFIX/POSTFIX)
/// @param k Number of final candidates to return
/// @return Tuple of vectors: 1st is sequences, 2nd is SW scores, 3rd is original IDs
std::tuple<std::vector<std::string>, std::vector<int>, std::vector<size_t>> sw_reranker(
    const std::vector<std::string> &cand_seqs,
    const std::vector<size_t> &cand_ids,
    const std::string &query_seq,
    size_t k);


/// @brief A L2 reranker that sort based on L2 distances between embeddings
std::pair<std::vector<std::string>, std::vector<float>> l2_reranker(const std::vector<std::string> &cand_seqs, const std::vector<float> &query_embedding, size_t k, Vectorizer &vectorizer);

/// @brief A batch reranker that takes in multiple queries and their candidates, and returns top k candidates for each query using L2 distance as metric
/// @param all_cand_seqs A vector of vector of candidate sequences for each query
/// @param all_neighbor_indices A flattened vector of all candidate indices across all queries. This helps to track which candidate belongs to which query.
/// @param query_start_indices A mapping vector indicating the start index of candidates for each query in all_cand_seqs. This help to track cand-query pairs.
/// @param query_embeddings A vector of query embeddings
/// @param k Number of final candidates to return for each query
/// @param vectorizer A reference to the Vectorizer object for embedding computation
/// @return List of tuples, each containing (sequences, L2 distances, original IDs) for each query
std::vector<std::tuple<std::vector<std::string>, std::vector<float>, std::vector<size_t>>> batch_reranker(
    const std::vector<std::string> &all_cand_seqs,
    const std::vector<size_t> &all_neighbor_indices,
    const std::vector<size_t> &query_start_indices,
    const std::vector<std::vector<float>> &query_embeddings,
    size_t k,
    Vectorizer &vectorizer);
