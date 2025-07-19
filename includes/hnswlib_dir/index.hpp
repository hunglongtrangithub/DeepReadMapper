#pragma once

#include <hnswlib/hnswlib.h>
#include "vectorize.hpp"
#include "config.hpp"
#include "utils.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

/// @brief Build an HNSW index from input data (using HNSWLib).
/// @param input_data A 2D array of input vectors, where each vector is a 1D array of floats.
/// @param index_file Path to save the built HNSW index file.
/// @param M Optional graph degree parameter (default from config)
/// @param EFC Optional construction parameter (default from config)
void build_index(const std::vector<std::vector<float>> &input_data, const std::string &index_file, int M = -1, int EFC = -1);
