#include "gann_hnsw.hpp"
#include "cnpy.h" // For saving numpy arrays
#include "util.hpp"
#include "vectorize.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <index_file> <query_file> <ef> <k>" << std::endl;
        std::cerr << "Example: " << argv[0] << " index.bin queries.txt 50 10" << std::endl;
        return 1;
    }

    std::string index_file = argv[1];
    std::string query_file = argv[2];
    int ef = std::stoi(argv[3]);
    int k = std::stoi(argv[4]);

    std::cout << "[SEARCH] Loading query file: " << query_file << std::endl;
    std::vector<std::string> sequences = read_file(query_file);

    // Embed input queries
    std::cout << "[SEARCH] Start inference for " << sequences.size() << " queries" << std::endl;
    Vectorizer vectorizer; // Use default params

    std::vector<std::vector<float>> embeddings = vectorizer.vectorize(sequences);
    std::cout << "[SEARCH] Inference completed" << std::endl;

    if (embeddings.empty())
    {
        std::cerr << "[SEARCH] Error: No query embeddings generated!" << std::endl;
        return 1;
    }

    // Load GANN-HNSW index
    std::cout << "[SEARCH] Loading GANN-HNSW index from " << index_file << "..." << std::endl;

    // Create index with dummy parameters (will be overwritten by load)
    GannHNSW index(embeddings[0].size(), 1000000, 16, 200);

    if (!index.load(index_file))
    {
        std::cerr << "[SEARCH] Error: Failed to load index from " << index_file << std::endl;
        return 1;
    }

    std::cout << "[SEARCH] Index loaded successfully!" << std::endl;
    std::cout << "[SEARCH] Index contains " << index.size() << " elements" << std::endl;
    std::cout << "[SEARCH] Search parameters: ef=" << ef << ", k=" << k << std::endl;

    std::cout << "[SEARCH] Start Parallel Search" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Perform search
    GannHNSW::SearchResult result;
    try
    {
        result = index.search(embeddings, k, ef);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[SEARCH] Error during search: " << e.what() << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[SEARCH] Search completed in " << duration.count() << " ms" << std::endl;

    // Convert flat results back to 2D format (similar to your original code)
    size_t num_queries = embeddings.size();
    std::vector<std::vector<uint32_t>> neighbors(num_queries, std::vector<uint32_t>(k));
    std::vector<std::vector<float>> distances(num_queries, std::vector<float>(k));

    for (size_t i = 0; i < num_queries; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            size_t idx = i * k + j;
            if (idx < result.ids.size())
            {
                neighbors[i][j] = result.ids[idx];
                distances[i][j] = result.distances[idx];
            }
        }
    }

    // Print sample results
    size_t cout_size = std::min(neighbors.size(), static_cast<size_t>(10));
    size_t cout_k = std::min(10, k); // Limit output to first 10 neighbors
    for (size_t i = 0; i < cout_size; ++i)
    {
        std::cout << "Query " << i << ": ";
        for (size_t j = 0; j < cout_k; ++j)
        {
            std::cout << "(" << neighbors[i][j] << ", " << distances[i][j] << ") ";
        }
        std::cout << std::endl;
    }

    // Save results as cnpy (matching your original format)
    std::cout << "[SEARCH] Saving results..." << std::endl;
    std::string indices_file = "indices.npy";
    std::string distances_file = "distances.npy";

    std::vector<uint32_t> host_indices(num_queries * k);
    std::vector<float> host_distances(num_queries * k);

    for (size_t i = 0; i < num_queries; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            host_indices[i * k + j] = neighbors[i][j];
            host_distances[i * k + j] = distances[i][j];
        }
    }

    // Save using cnpy with configurable file names
    cnpy::npy_save(indices_file, host_indices.data(), {static_cast<unsigned long>(num_queries), static_cast<unsigned long>(k)});
    cnpy::npy_save(distances_file, host_distances.data(), {static_cast<unsigned long>(num_queries), static_cast<unsigned long>(k)});

    std::cout << "[SEARCH] Results saved to " << indices_file << " and " << distances_file << std::endl;
    std::cout << "[SEARCH] Search process completed!" << std::endl;

    return 0;
}