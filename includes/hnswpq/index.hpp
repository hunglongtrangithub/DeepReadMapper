#pragma once

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <chrono>
#include <vector>
#include <omp.h>
#include "utils.hpp"
#include "vectorize.hpp"
#include "config.hpp"

/// @brief Create a representative training set by sampling evenly across the entire dataset.
/// Takes n_train vectors, usually 10-20% of total for optimal PQ codebook quality.
/// @param all_embeddings The complete set of embeddings to sample from
/// @param n_train Number of training vectors to sample
/// @return Flattened training data suitable for FAISS training
std::vector<float> create_training_set(
    const std::vector<std::vector<float>> &all_embeddings,
    int n_train);

/// @brief Build a FAISS IndexHNSWPQ index from input embeddings.
/// @param input_data 2D vector of embeddings, where each inner vector is one embedding
/// @param index_file Output file path to save the built index
/// @param M_pq Number of PQ subquantizers (default: 8, must divide dimension)
/// @param nbits Bits per subquantizer (default: 8, must be 8, 10, or 12)
/// @param M_hnsw HNSW connectivity parameter (default: 16)
/// @param EFC efConstruction parameter for HNSW building (default: 200)
void build_faiss_index(const std::vector<std::vector<float>> &input_data,
                       const std::string &index_file,
                       int M_pq = 8, 
                       int nbits = 8, 
                       int M_hnsw = 16, 
                       int EFC = 200);