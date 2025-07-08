#include "hnsw.h"
#include <iostream>
#include <vector>
#include <random>

int main() {
    // Parameters for HNSW
    uint32_t dim = 128;           // Vector dimension (matches VECTOR_DIM)
    uint32_t efc = 200;           // ef construction parameter
    uint32_t M = 16;              // Number of connections per node
    uint64_t maxNumNodes = 100000; // Maximum number of nodes
    
    // Create HNSW index
    HNSW hnsw(dim, efc, M, maxNumNodes);
    
    // Generate some sample data (replace with your actual data)
    std::vector<std::vector<float>> vectors;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    int numVectors = 10000;
    vectors.reserve(numVectors);
    
    std::cout << "Generating " << numVectors << " random vectors..." << std::endl;
    for (int i = 0; i < numVectors; i++) {
        std::vector<float> vec(dim);
        // Fix: use uint32_t for j to match dim type
        for (uint32_t j = 0; j < dim; j++) {
            vec[j] = dis(gen);
        }
        vectors.push_back(vec);
    }
    
    // Build the index
    std::cout << "Building HNSW index..." << std::endl;
    hnsw.buildIndex(vectors);
    
    // Print summary
    hnsw.summarize();
    
    // Save index to file
    std::string filename = "hnsw_index.bin";
    std::cout << "Saving index to " << filename << "..." << std::endl;
    hnsw.save(filename);
    
    std::cout << "Index saved successfully!" << std::endl;
    
    // Example: Load the index back and test search
    std::cout << "Loading index back..." << std::endl;
    HNSW loaded_hnsw = loadHNSW(filename);
    loaded_hnsw.summarize();
    
    // Test search with a query vector
    std::vector<float> query(dim);
    // Fix: use uint32_t for i to match dim type
    for (uint32_t i = 0; i < dim; i++) {
        query[i] = dis(gen);
    }
    
    uint32_t ef = 50; // Search parameter
    auto results = loaded_hnsw.search(query, ef);
    
    std::cout << "Search results (top 5):" << std::endl;
    for (int i = 0; i < std::min(5, (int)results.size()); i++) {
        std::cout << "  Node " << results[i].second 
                  << " (distance: " << results[i].first << ")" << std::endl;
    }
    
    return 0;
}