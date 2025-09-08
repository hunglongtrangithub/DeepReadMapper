#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <utility>
#include <numeric>
#include "vectorize.hpp"
#include "metrics.hpp"

/// @brief A reranker that takes in EF * 2 * stride candidates and returns top k candidates as (seq, dists) using Smith-Waterman as alignment metric
/// @param cand_seqs Vector of candidate sequences (removed PREFIX/POSTFIX)
/// @param query_seq Query sequence to align against (removed PREFIX/POSTFIX)
/// @param k Number of final candidates to return
/// @param stride Stride used during FASTA preprocessing
/// @return Pair of vectors: 1st is labels, 2nd is edit distances (SW-score)
std::pair<std::vector<std::string>, std::vector<int>> reranker(const std::vector<std::string> &cand_seqs, const std::string &query_seq, size_t k);

/// @brief An overload reranker that sort based on L2 distances between embeddings
std::pair<std::vector<std::string>, std::vector<float>> reranker(const std::vector<std::string> &cand_seqs, const std::vector<float> &query_embedding, size_t k, Vectorizer &vectorizer);

/// @brief A batch reranker that takes in multiple queries and their candidates, and returns top k candidates for each query using L2 distance as metric
/// @param all_cand_seqs A vector of vector of candidate sequences for each query
/// @param query_embeddings A vector of query embeddings
/// @param k Number of final candidates to return for each query
/// @param vectorizer A reference to the Vectorizer object for embedding computation
/// @return 
std::vector<std::pair<std::vector<std::string>, std::vector<float>>> batch_reranker(
    const std::vector<std::vector<std::string>> &all_cand_seqs, 
    const std::vector<std::vector<float>> &query_embeddings, 
    size_t k, 
    Vectorizer &vectorizer);