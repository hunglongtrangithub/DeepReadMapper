#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

/// @brief Read input sequences from a text file. This file is formatted FASTA/FASTQ, where each line is a sequence.
/// @param file_path Path to the input file.
std::vector<std::string> read_file_default(const std::string &file_path);

#ifdef __linux__
/// @brief Read input sequences using memory mapping (Linux only).
/// @param file_path Path to the input file.
/// @return Vector of input sequences as strings.
std::vector<std::string> read_file_mmap(const std::string &file_path);
#endif

// Add this to your header:
/// @brief Wrapper function for reading input sequences efficiently.
/// @param file_path Path to the input file
/// @return Vector of input sequences as strings  
std::vector<std::string> read_file(const std::string &file_path);

/// @brief Analyze input sequences and print statistics.
/// @param sequences Vector of input sequences as strings.
void analyze_input(const std::vector<std::string> &sequences);

