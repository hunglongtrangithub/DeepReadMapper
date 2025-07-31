#include <chrono>
#include <iostream>
#include "cnpy.h"
#include "hnswlib_dir/search.hpp"
#include "util.hpp"
#include "vectorize.hpp"

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <index_file> <query_file> <ef> <k>" << std::endl;
        return 1;
    }

    std::string index_file = argv[1];
    std::string query_file = argv[2];
    int ef = std::stoi(argv[3]);
    int k = std::stoi(argv[4]);

    std::cout << "[MAIN] Loading query file: " << query_file << std::endl;
    std::vector<std::string> sequences = read_file(query_file);

    // Embed input queries
    std::cout << "[MAIN] Start inference" << std::endl;
    Vectorizer vectorizer; // Use default params

    std::vector<std::vector<float>> embeddings = vectorizer.vectorize(sequences);
    std::cout << "[MAIN] Inference completed" << std::endl;

    // Load HNSW index
    std::cout << "[MAIN] Loading HNSW index from " << index_file << "..." << std::endl;
    hnswlib::L2Space space(Config::Build::DIM);
    hnswlib::HierarchicalNSW<float> *index = new hnswlib::HierarchicalNSW<float>(&space, index_file);

    std::cout << "[MAIN] Index loaded successfully!" << std::endl;

    std::cout << "[MAIN] Start Parallel Search" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Perform search
    auto [neighbors, distances] = search(index, embeddings, k, ef);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[MAIN] Search completed in " << duration.count() << " ms" << std::endl;

    size_t cout_size = std::min(neighbors.size(), static_cast<size_t>(10));
    size_t cout_k = std::min(10, static_cast<int>(neighbors[0].size())); // Limit output to first 10 neighbors
    for (size_t i = 0; i < cout_size; ++i)
    {
        std::cout << "Query " << i << ": ";
        for (size_t j = 0; j < cout_k; ++j)
        {
            std::cout << "(" << neighbors[i][j] << ", " << distances[i][j] << ") ";
        }
        std::cout << std::endl;
    }

    // Save results as cnpy
    std::cout << "[MAIN] Saving results..." << std::endl;
    std::string indices_file = "indices.npy";
    std::string distances_file = "distances.npy";
    size_t n_rows = neighbors.size();
    std::vector<uint32_t> host_indices(n_rows * k);
    std::vector<float> host_distances(n_rows * k);

    for (size_t i = 0; i < n_rows; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            host_indices[i * k + j] = neighbors[i][j];
            host_distances[i * k + j] = distances[i][j];
        }
    }

    // Save using cnpy with configurable file names
    cnpy::npy_save(indices_file, host_indices.data(), {static_cast<unsigned long>(n_rows), static_cast<unsigned long>(k)});
    cnpy::npy_save(distances_file, host_distances.data(), {static_cast<unsigned long>(n_rows), static_cast<unsigned long>(k)});

    std::cout << "[MAIN] Results saved to " << indices_file << " and " << distances_file << std::endl;

    std::cout <<"[MAIN] Finish search" << std::endl;

    delete index; // Clean up
    return 0;
}