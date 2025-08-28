#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <stdexcept>
#include <cctype>
#include <filesystem> // C++17
#include "utils.hpp"

#define PREFIX "<"
#define POSTFIX ">"

/// @brief Estimate the number of tokens that will be generated from a FASTA file
/// @param fasta_path Path to the FASTA file
/// @param token_len Length of each token/window
/// @return Estimated number of tokens (including forward and reverse complement)
size_t estimate_token_count(const std::string &fasta_path, int token_len);

/// @brief Compute the reverse complement of a DNA sequence
/// @param seq Input DNA sequence string
/// @return Reverse complement of the input sequence
std::string reverse_complement(const std::string &seq);

/// @brief Read FASTA file using traditional file I/O
/// @param fasta_file Path to the FASTA file
/// @param buffer Buffer to store file data (for default I/O)
/// @return Pair of data pointer and size
std::pair<const char*, size_t> read_fasta_default(const std::string &fasta_file, std::unique_ptr<char[]> &buffer);

/// @brief Read FASTA file using memory mapping (Linux only)
/// @param fasta_file Path to the FASTA file
/// @param fd File descriptor (for cleanup)
/// @return Pair of data pointer and size
std::pair<const char*, size_t> read_fasta_mmap(const std::string &fasta_file, int &fd);

/// @brief Wrapper function for FASTA file reading
/// @param fasta_file Path to the FASTA file
/// @param buffer Buffer to store file data (for default I/O)
/// @param fd File descriptor (for mmap cleanup)
/// @return Pair of data pointer and size
std::pair<const char*, size_t> read_fasta(const std::string &fasta_file, std::unique_ptr<char[]> &buffer, int &fd);

/// @brief Process FASTA data (single-threaded, separated from I/O)
/// @param data Pointer to file data
/// @param data_size Size of data
/// @param fasta_file Original filename (for estimation)
/// @param ref_len Length of each reference sequence to cut into (doesn't include PREFIX/POSTFIX)
/// @return Vector of formatted sequences
std::vector<std::string> format_fasta(const char *data, size_t data_size, const std::string &fasta_file, int ref_len);

/// @brief Combined FASTA formatting with I/O handling
/// @param fasta_file Path to the FASTA file
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX)
/// @return Vector of formatted sequences
std::vector<std::string> preprocess_fasta(const std::string &fasta_file, int ref_len);

/// @brief Wrapper function for FASTQ preprocessing that chooses optimal method
/// @param fastq_file Path to the FASTQ file
/// @return Vector of formatted sequences with PREFIX and POSTFIX tags
std::vector<std::string> preprocess_fastq(const std::string &fastq_file);