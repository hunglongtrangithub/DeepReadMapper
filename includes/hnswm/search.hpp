#pragma once

#include <vector>

#include "hnsw.h"

/// @brief Search for nearest neighbors of query vectors in an HNSW index.
/// @param quer_vecs A 2D vector where each inner vector is a query vector.
/// @param index HNSW index
/// @return A pair containing 2 vectors (1D array): - 1st contains node IDs & 2nd contains corresponding distances.
std::pair<std::vector<std::vector<uint32_t>>, std::vector<std::vector<float>>> search(
    const std::vector<std::vector<float>>& quer_vecs, HNSW& index);
