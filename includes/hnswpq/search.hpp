#pragma once

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <omp.h>

#include <chrono>

#include "cnpy.h"
#include "config.hpp"
#include "utils.hpp"
#include "vectorize.hpp"

/// @brief Parallel search for nearest neighbors in an HNSWPQ index using FAISS and OpenMP.
/// @param index A FAISS IndexHNSWPQ object.
/// @param query_data  A 2D array of query vectors, where each vector is a 1D array of floats.
/// @param k Number of nearest neighbors to return (default: from Config::Search::K)
/// @param ef Search parameter for HNSW (default: from Config::Search::EF)
/// @return A pair of 1D arrays, 1st contain neighbor_ids, 2nd contain respective distances.
std::pair<std::vector<std::vector<faiss::idx_t>>, std::vector<std::vector<float>>> faiss_search(
    faiss::IndexHNSWPQ *index, const std::vector<std::vector<float>> &query_data, int k = Config::Search::K,
    int ef = Config::Search::EF);