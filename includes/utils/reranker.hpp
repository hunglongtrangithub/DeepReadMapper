#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <utility>
#include <numeric>
#include "vectorize.hpp"
#include "metrics.hpp"

// TODO: Implement Banded Smith-Waterman for faster reranking
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

/// @brief A L2 reranker that sort based on L2 distances between embeddings, for 1 query

std::pair<std::vector<std::string>, std::vector<float>> l2_reranker(const std::vector<std::string> &cand_seqs, const std::vector<float> &query_embedding, size_t k, Vectorizer &vectorizer);

/// @brief Batch reranker with embedding indices using L2 distance, process all queries in batch
/// @param cand_seqs All candidate sequences
/// @param dense_ids Dense IDs corresponding to cand_seqs
/// @param cand_embedding_ids Indices into cand_embeddings for each candidate
/// @param cand_embeddings Reference to candidate embedding pool
/// @param query_start_ids Start indices for each query's candidates
/// @param query_embeddings Query embeddings
/// @param k Number of top results to return
/// @return Vector of tuples (top_seqs, top_dists, top_ids) for each query
std::vector<std::tuple<std::vector<std::string>, std::vector<float>, std::vector<size_t>>> batch_reranker(
    const std::vector<std::string> &cand_seqs,
    const std::vector<size_t> &dense_ids,
    const std::vector<size_t> &cand_embedding_ids,
    const std::vector<std::vector<float>> &cand_embeddings,
    const std::vector<size_t> &query_start_ids,
    const std::vector<std::vector<float>> &query_embeddings,
    size_t k);
