#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#endif

///@brief AVX2 optimized Smith-Waterman alignment score computation
///@param seq1 First sequence
///@param seq2 Second sequence
///@return Smith-Waterman alignment score
int calc_sw_score_avx2(const std::string &seq1, const std::string &seq2);

///@brief Compute the Smith-Waterman alignment score between two sequences
///@param seq1 First sequence
///@param seq2 Second sequence
///@return Smith-Waterman alignment score
int calc_sw_score(const std::string &seq1, const std::string &seq2);

/// @brief Calculate L2 distance between two vectors (or euclidean distance)
/// @param vec1 First vector
/// @param vec2 Second vector
/// @return L2 distance
float calc_l2_dist(const std::vector<float> &vec1, const std::vector<float> &vec2);
