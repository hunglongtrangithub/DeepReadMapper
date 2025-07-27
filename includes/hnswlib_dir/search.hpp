#pragma once

#include <hnswlib/hnswlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <chrono>
#include "cnpy.h"
#include "utils.hpp"
#include "config.hpp"
#include "vectorize.hpp"

/// @brief Parallel search for nearest neighbors in an HNSW index using HNSWLib and OpenMP.
/// @param index A HNSW index object.
/// @param query_data  A 2D array of query vectors, where each vector is a 1D array of floats.
/// @param k Number of nearest neighbors to return (default: from Config::Search::K)
/// @param ef Search parameter for HNSW (default: from Config::Search::EF)
/// @return A pair of 1D arrays, 1st contain neighbor_ids, 2nd contain respective distances.
std::pair<std::vector<std::vector<hnswlib::labeltype>>, std::vector<std::vector<float>>> search(
    hnswlib::HierarchicalNSW<float> *index,
    const std::vector<std::vector<float>> &query_data,
    int k = Config::Search::K,
    int ef = Config::Search::EF);