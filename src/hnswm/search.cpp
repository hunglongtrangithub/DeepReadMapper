#include "hnswm/search.hpp"

std::pair<std::vector<std::vector<uint32_t>>, std::vector<std::vector<float>>> search(const std::vector<std::vector<float>> &quer_vecs, HNSW &index)
{
    std::vector<std::vector<uint32_t>> node_ids(quer_vecs.size());
    std::vector<std::vector<float>> distances(quer_vecs.size());

    // Define Search Config
    uint32_t EF = Config::Search::EF;

    std::cout << "[SEARCH] Start search for " << quer_vecs.size() << " queries" << std::endl;

    auto batch_results = index.searchParallel(quer_vecs, EF);

    std::cout << "[SEARCH] Search completed" << std::endl;

    // Reformat output data from std::vector<std::vector<search_result_t>>
    for (size_t i = 0; i < batch_results.size(); ++i)
    {
        const auto &query_results = batch_results[i];

        for (const auto &result : query_results)
        {
            distances[i].push_back(result.first); // float distance
            node_ids[i].push_back(result.second); // nodeID_t (uint32_t)
        }
    }

    return {node_ids, distances};
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <quer_file.txt> <search.index>" << std::endl;
        return 1;
    }

    std::string query_file = argv[1];
    std::string index_file = argv[2];

    // Load query file
    std::cout << "[MAIN] Loading query file: " << query_file << std::endl;
    std::vector<std::string> sequences = read_file(query_file);

    // Embed input queries
    std::cout << "[MAIN] Start inference" << std::endl;
    Vectorizer vectorizer; // Use default params

    std::vector<std::vector<float>> quer_vecs = vectorizer.vectorize(sequences);
    std::cout << "[MAIN] Inference completed" << std::endl;

    // Load existing index
    std::cout << "[MAIN] Loading HNSW index from " << index_file << "..." << std::endl;

    HNSW index = loadHNSW(index_file);
    std::cout << "[MAIN] Index loaded successfully!" << std::endl;
    index.summarize();

    // Test batch parallel search
    std::cout << "[MAIN] Start Parallel Search" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto [neighbors, distances] = search(quer_vecs, index);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    size_t numQueries = quer_vecs.size();
    std::cout << "[MAIN] Parallel search for " << numQueries << " queries took: "
              << duration.count() << " ms" << std::endl;
    std::cout << "Average per query: " << (duration.count() * 1000.0 / numQueries) << " μs" << std::endl;

    std::cout << "[MAIN] Save results to file..." << std::endl;

    // Convert 2D vectors to 1D flattened arrays (same format as main.cu)
    size_t n_rows = neighbors.size();
    size_t k = Config::Search::K;

    // Flatten the 2D vectors into 1D arrays
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

    // Save using cnpy
    std::string indices_file = "indices.npy";
    std::string distances_file = "distances.npy";

    cnpy::npy_save(indices_file, host_indices.data(), {static_cast<unsigned long>(n_rows), static_cast<unsigned long>(k)});

    cnpy::npy_save(distances_file, host_distances.data(), {static_cast<unsigned long>(n_rows), static_cast<unsigned long>(k)});

    std::cout << "[MAIN] Results saved to " << indices_file << " and " << distances_file << std::endl;
    std::cout << "[MAIN] Parallel Search completed!" << std::endl;

    return 0;
}