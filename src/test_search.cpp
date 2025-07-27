#include "hnswlib_dir/search.hpp"

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

    delete index; // Clean up
    return 0;
}