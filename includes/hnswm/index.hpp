#pragma once

#include <string>
#include <vector>

/// @brief Build an HNSW index from a given vector lists and save it to an index file.
/// @param ref_vecs A 2D vector where each inner vector is a reference vector
/// @param index_file
void build(const std::vector<std::vector<float>> ref_vecs, std::string index_file);
