#pragma once

#include "hnsw.h"
#include "utils.hpp"
#include "vectorize.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

/// @brief Build an HNSW index from a given vector lists and save it to an index file.
/// @param ref_vecs A 2D vector where each inner vector is a reference vector
/// @param index_file
void build(std::vector<std::vector<float>> ref_vecs, std::string index_file);
