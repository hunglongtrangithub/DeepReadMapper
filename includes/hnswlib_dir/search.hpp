#pragma once

#include <hnswlib/hnswlib.h>
#include "config.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <omp.h>

/// @brief Parallel search for nearest neighbors in an HNSW index using HNSWLib and OpenMP.
/// @param index A HNSW index object.
/// @param query_data  A 2D array of query vectors, where each vector is a 1D array of floats.
/// @return A pair of 1D arrays, 1st contain neighbor_ids, 2nd contain respective distances.
std::pair<std::vector<std::vector<hnswlib::labeltype>>, std::vector<std::vector<float>>> search(hnswlib::HierarchicalNSW<float> *index, const std::vector<std::vector<float>> &query_data);
