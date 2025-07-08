#pragma once

#include "hnswlib.h"
#include "vectorize.hpp"
#include "config.hpp"
#include "utils.hpp"
#include <iostream>
#include <vector>
#include <string>

/// @brief Build an HNSW index from input data (using HNSWLib).
/// @param input_data A 2D array of input vectors, where each vector is a 1D array of floats.
/// @param index_file Path to save the built HNSW index file.
void build_index(const std::vector<std::vector<float>> &input_data, const std::string &index_file);
