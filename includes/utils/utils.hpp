#pragma once

#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem> // C++17
#include "cnpy.h"

/// @brief Read input sequences from a text file. This file is formatted FASTA/FASTQ, where each line is a sequence.
/// @param file_path Path to the input file.
std::vector<std::string> read_txt_default(const std::string &file_path);

#ifdef __linux__
/// @brief Read input sequences using memory mapping (Linux only).
/// @param file_path Path to the input file.
/// @return Vector of input sequences as strings.
std::vector<std::string> read_txt_mmap(const std::string &file_path);
#endif

// Add this to your header:
/// @brief Wrapper function for reading input sequences efficiently.
/// @param file_path Path to the input file
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX (for FASTA only).
/// @return Vector of input sequences as strings
std::vector<std::string> read_file(const std::string &file_path, int ref_len = 150);

/// @brief Analyze input sequences and print statistics.
/// @param sequences Vector of input sequences as strings.
void analyze_input(const std::vector<std::string> &sequences);

/// @brief Save search results to files.
/// @param neighbors Vector of vectors containing neighbor indices.
/// @param distances Vector of vectors containing distances.
/// @param indices_file Output file for neighbor indices.
/// @param distances_file Output file for distances.
/// @param k Number of nearest neighbors to save.
/// @param use_npy Whether to save results in .npy format (default: true).
/// @return 0 if successful.
int save_results(const std::vector<std::vector<size_t>> &neighbors, const std::vector<std::vector<float>> &distances, const std::string &indices_file, const std::string &distances_file, size_t k, const bool use_npy = true);

/// @brief Overload of save_results to handle faiss::idx_t type for neighbors. ALl params are the same as above.
/// @param neighbors
/// @param distances
/// @param indices_file
/// @param distances_file
/// @param k
/// @param use_npy
/// @return
int save_results(const std::vector<std::vector<long int>> &neighbors, const std::vector<std::vector<float>> &distances, const std::string &indices_file, const std::string &distances_file, size_t k, const bool use_npy = true);