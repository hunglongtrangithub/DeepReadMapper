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
#include <variant>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <atomic>

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

/// @brief Generic file reader (auto-detects format)
/// @return Pair of (sequences, query IDs). For FASTA/TXT returns empty IDs, for FASTQ returns actual IDs
std::pair<std::vector<std::string>, std::vector<std::string>> read_file(const std::string &file_path, size_t ref_len = 150, size_t stride = 1, bool lookup_mode = false);

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
//! This function is deprecated as we return SAM instead of bin/npy files

int save_results(const std::vector<std::vector<size_t>> &neighbors, const std::vector<std::vector<float>> &distances, const std::string &indices_file, const std::string &distances_file, size_t k, const bool use_npy = true);

/// @brief Overload of save_results to handle faiss::idx_t type for neighbors. ALl params are the same as above.
/// @param neighbors
/// @param distances
/// @param indices_file
/// @param distances_file
/// @param k
/// @param use_npy
/// @return
//! This function is deprecated as we return SAM instead of bin/npy files

int save_results(const std::vector<std::vector<long int>> &neighbors, const std::vector<std::vector<float>> &distances, const std::string &indices_file, const std::string &distances_file, size_t k, const bool use_npy = true);

/// @brief Write results to a SAM file. Normalize position to 1-indexed.
/// @param final_seqs Vector of final aligned sequences.
/// @param final_scores Vector of final alignment scores.
/// @param query_seqs Vector of original query sequences.
/// @param sequence_ids Vector of original sequence IDs corresponding to final_seqs/final_scores.
/// @param ref_name Name of the reference sequence.
/// @param ref_len Length of the reference sequence.
/// @param k Number of alignments per query.
/// @param output_file Output SAM file path.
void write_sam(const std::vector<std::string>& final_seqs, 
               const std::vector<float>& final_scores,
               const std::vector<std::string>& query_seqs,
               const std::vector<std::string>& query_ids,      // Query IDs from FASTQ headers
               const std::vector<size_t>& sequence_ids,        // Candidate sequence IDs
               const std::string& ref_name,
               const int ref_len,
               size_t k, 
               const std::string& output_file);

/// @brief Write SAM output in streaming mode (appends to file)
void write_sam_streaming(
    const std::vector<std::string> &cand_seqs,
    const std::vector<float> &scores,
    const std::vector<std::string> &query_seqs,
    const std::vector<std::string> &query_ids,
    const std::vector<size_t> &cand_ids,
    const std::string &ref_name,
    size_t ref_len,
    size_t k,
    const std::string &sam_file,
    size_t query_offset = 0,
    bool write_header = false,
    size_t batch_query_count = 0
    );

using ConfigValue = std::variant<size_t, float, std::string>;

/// @brief Save index config to a text file.
/// @param config Unordered map of config parameters.
/// @param folder_path Folder path to save config file.
/// @param config_file Config file name (default: "config.txt").
/// @return 0 if successful.
int save_config(const std::unordered_map<std::string, ConfigValue> &config, const std::string &folder_path, const std::string &config_file = "config.txt");

/// @brief Load index config from a text file.
/// @param config_file Input config file path.
/// @return Unordered map of config parameters.
/// @example {"dim": 128, "M_pq": 16, "nbits": 8, "M_hnsw": 32, "EFC": 200, "EF": 50, "K": 10, "index_type": "HNSWPQ"}
std::unordered_map<std::string, ConfigValue> load_config(const std::string &config_file);

/// @brief Save custom label mapping to a binary file.
/// @param labels Vector of labels to save.
/// @param folder_path Folder path to save mapping file.
/// @param mapping_file Mapping file name
/// @return 0 if successful.
int save_id_map(const std::vector<size_t> &labels, const std::string &folder_path, const std::string &mapping_file = "id_map.bin");

/// @brief Load custom label mapping from a binary file.
/// @param folder_path Folder path where mapping file is located.
/// @param mapping_file Mapping file name
/// @return Vector of labels loaded from file.
std::vector<size_t> load_id_map(const std::string &mapping_file = "id_map.bin");
