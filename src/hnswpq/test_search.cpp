#include "hnswpq/search.hpp"
#include "cnpy.h"
#include <filesystem>

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

    std::vector<std::vector<float>> embeddings;

    // Check if query file is .npy
    std::string file_ext = std::filesystem::path(query_file).extension().string();
    
    if (file_ext == ".npy")
    {
        std::cout << "[MAIN] Loading embeddings directly from .npy file: " << query_file << std::endl;
        
        // Load .npy file
        cnpy::NpyArray arr = cnpy::npy_load(query_file);
        
        if (arr.shape.size() != 2)
        {
            std::cerr << "Error: Expected 2D array in .npy file" << std::endl;
            return 1;
        }
        
        size_t num_queries = arr.shape[0];
        size_t embedding_dim = arr.shape[1];
        
        std::cout << "[MAIN] Loaded " << num_queries << " embeddings of dimension " << embedding_dim << std::endl;
        
        // Convert to vector<vector<float>>
        float* data = arr.data<float>();
        embeddings.resize(num_queries);
        for (size_t i = 0; i < num_queries; ++i)
        {
            embeddings[i].resize(embedding_dim);
            for (size_t j = 0; j < embedding_dim; ++j)
            {
                embeddings[i][j] = data[i * embedding_dim + j];
            }
        }
    }
    else
    {
        std::cout << "[MAIN] Loading query file: " << query_file << std::endl;
        auto [sequences, _] = read_file(query_file);

        // Embed input queries
        std::cout << "[MAIN] Start inference" << std::endl;
        Vectorizer vectorizer; // Use default params

        embeddings = vectorizer.vectorize(sequences);
        std::cout << "[MAIN] Inference completed" << std::endl;
    }

    // Load FAISS IndexHNSWPQ
    std::cout << "[MAIN] Loading FAISS IndexHNSWPQ from " << index_file << "..." << std::endl;
    faiss::Index *loaded_index = faiss::read_index(index_file.c_str());
    faiss::IndexHNSWPQ *index = dynamic_cast<faiss::IndexHNSWPQ *>(loaded_index);

    if (!index)
    {
        std::cerr << "Error: Failed to load IndexHNSWPQ from " << index_file << std::endl;
        delete loaded_index;
        return 1;
    }

    std::cout << "[MAIN] Index loaded successfully!" << std::endl;
    std::cout << "[MAIN] Index contains " << index->ntotal << " vectors" << std::endl;

    // Set multi-threading for FAISS
    omp_set_num_threads(Config::Search::NUM_THREADS);

    std::cout << "[MAIN] Start Parallel Search (efSearch=" << ef << ")" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Perform search
    auto [neighbors, distances] = faiss_search(index, embeddings, k, ef);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[MAIN] Search completed in " << duration.count() << " ms" << std::endl;

    // Display sample results
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
        for (size_t j = 0; j < static_cast<size_t>(k); ++j)
        {
            host_indices[i * k + j] = static_cast<uint32_t>(neighbors[i][j]);
            host_distances[i * k + j] = distances[i][j];
        }
    }

    // Save using cnpy with configurable file names
    cnpy::npy_save(indices_file, host_indices.data(), {static_cast<unsigned long>(n_rows), static_cast<unsigned long>(k)});
    cnpy::npy_save(distances_file, host_distances.data(), {static_cast<unsigned long>(n_rows), static_cast<unsigned long>(k)});

    std::cout << "[MAIN] Results saved to " << indices_file << " and " << distances_file << std::endl;

    std::cout << "[MAIN] Finish search" << std::endl;

    delete index; // Clean up
    return 0;
}