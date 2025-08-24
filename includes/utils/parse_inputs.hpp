#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <stdexcept>
#include <cctype>
#include <filesystem> // C++17
#include "utils.hpp"

#define REFERENCE_SEQ_LEN 150 // does not include prefix/postfix
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

/// @brief Parse FASTA file and extract sequences with sliding window approach
/// @param fasta_file Path to the FASTA file
/// @return Vector of formatted sequences with PREFIX and POSTFIX tags, including reverse complements
std::vector<std::string> preprocess_fasta(const std::string &fasta_file);

/// @brief Parse FASTQ file and extract sequences
/// @param fastq_file Path to the FASTQ file
/// @return Vector of formatted sequences with PREFIX and POSTFIX tags
std::vector<std::string> preprocess_fastq(const std::string &fastq_file);