#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdexcept>

/// @brief Read input sequences from a text file. This file is formatted FASTA/FASTQ, where each line is a sequence.
/// @param file_path Path to the input file.
std::vector<std::string> read_file(const std::string &file_path);

/// @brief Analyze input sequences and print statistics.
/// @param sequences Vector of input sequences as strings.
void analyze_input(const std::vector<std::string> &sequences);
