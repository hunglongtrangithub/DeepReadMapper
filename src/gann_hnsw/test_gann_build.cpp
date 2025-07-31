#include "gann_hnsw.hpp"
#include "utils.hpp"
#include "vectorize.hpp"
#include <iostream>
#include <chrono>
#include <string>
#include <vector>

// You'll need to include your existing utility functions
// #include "your_vectorizer.hpp"  // For Vectorizer class
// #include "your_file_utils.hpp"  // For read_file function

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " <ref_file> <index_file> <M> <ef_construction> <num_threads>" << std::endl;
        std::cerr << "Example: " << argv[0] << " data.txt index.bin 16 200 8" << std::endl;
        return 1;
    }

    std::string ref_file = argv[1];
    std::string index_file = argv[2];
    int M = std::stoi(argv[3]);
    int ef_construction = std::stoi(argv[4]);
    int num_threads = std::stoi(argv[5]);

    std::cout << "[BUILD] Loading data file: " << ref_file << std::endl;
    
    // Load your data (adjust this based on your data format)
    std::vector<std::string> sequences = read_file(ref_file);
    
    // Embed input data
    std::cout << "[BUILD] Start inference for " << sequences.size() << " sequences" << std::endl;
    Vectorizer vectorizer; // Use default params
    
    std::vector<std::vector<float>> embeddings = vectorizer.vectorize(sequences);
    std::cout << "[BUILD] Inference completed" << std::endl;
    
    if (embeddings.empty()) {
        std::cerr << "[BUILD] Error: No embeddings generated!" << std::endl;
        return 1;
    }
    
    size_t dimension = embeddings[0].size();
    size_t num_elements = embeddings.size();
    
    std::cout << "[BUILD] Data info:" << std::endl;
    std::cout << "  - Number of vectors: " << num_elements << std::endl;
    std::cout << "  - Dimension: " << dimension << std::endl;
    std::cout << "  - M: " << M << std::endl;
    std::cout << "  - ef_construction: " << ef_construction << std::endl;
    std::cout << "  - num_threads: " << num_threads << std::endl;

    // Create GANN-HNSW index
    std::cout << "[BUILD] Creating GANN-HNSW index..." << std::endl;
    GannHNSW index(dimension, num_elements, M, ef_construction);

    // Build index
    std::cout << "[BUILD] Building index..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        index.build(embeddings, num_threads);
    } catch (const std::exception& e) {
        std::cerr << "[BUILD] Error building index: " << e.what() << std::endl;
        return 1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[BUILD] Index built in " << duration.count() << " ms" << std::endl;

    // Save index
    std::cout << "[BUILD] Saving index to " << index_file << "..." << std::endl;
    try {
        index.save(index_file);
        std::cout << "[BUILD] Index saved successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[BUILD] Error saving index: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "[BUILD] Build process completed!" << std::endl;
    std::cout << "[BUILD] Index contains " << index.size() << " elements" << std::endl;
    
    return 0;
}