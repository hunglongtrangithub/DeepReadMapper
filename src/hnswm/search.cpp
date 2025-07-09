#include "hnsw.h"
#include "bruteforce.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

int main() {
    // Load existing index
    std::string filename = "hnsw_index.bin";
    std::cout << "Loading HNSW index from " << filename << "..." << std::endl;
    
    // Fix: Cannot declare HNSW without constructor parameters
    try {
        HNSW hnsw = loadHNSW(filename);  // Move declaration inside try block
        std::cout << "Index loaded successfully!" << std::endl;
        hnsw.summarize();
        
        // Generate test queries
        uint32_t dim = 128;
        int numQueries = 100;
        std::vector<std::vector<float>> queries;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        std::cout << "Generating " << numQueries << " test queries..." << std::endl;
        queries.reserve(numQueries);
        for (int i = 0; i < numQueries; i++) {
            std::vector<float> query(dim);
            // Fix: use uint32_t for j to match dim type
            for (uint32_t j = 0; j < dim; j++) {
                query[j] = dis(gen);
            }
            queries.push_back(query);
        }
        
        // Test single query search
        std::cout << "\n=== Single Query Search Test ===" << std::endl;
        uint32_t ef = 50;
        uint32_t k = 10;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto results = hnsw.search(queries[0], ef);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Single search took: " << duration.count() << " μs" << std::endl;
        std::cout << "Found " << results.size() << " results:" << std::endl;
        
        // Fix: Cast k to int to match comparison types
        for (int i = 0; i < std::min((int)k, (int)results.size()); i++) {
            std::cout << "  Rank " << (i+1) << ": Node " << results[i].second 
                      << " (distance: " << results[i].first << ")" << std::endl;
        }
        
        // Test batch parallel search
        std::cout << "\n=== Parallel Batch Search Test ===" << std::endl;
        
        start = std::chrono::high_resolution_clock::now();
        auto batch_results = hnsw.searchParallel(queries, ef);
        end = std::chrono::high_resolution_clock::now();
        
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Parallel search for " << numQueries << " queries took: " 
                  << duration.count() << " ms" << std::endl;
        std::cout << "Average per query: " << (duration.count() * 1000.0 / numQueries) 
                  << " μs" << std::endl;
        
        // Test sequential search for comparison
        std::cout << "\n=== Sequential Search Comparison ===" << std::endl;
        
        start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<search_result_t>> sequential_results;
        for (const auto& query : queries) {
            sequential_results.push_back(hnsw.search(query, ef));
        }
        end = std::chrono::high_resolution_clock::now();
        
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Sequential search for " << numQueries << " queries took: " 
                  << duration.count() << " ms" << std::endl;
        std::cout << "Average per query: " << (duration.count() * 1000.0 / numQueries) 
                  << " μs" << std::endl;
        
        // Test different ef values
        std::cout << "\n=== Search Quality vs Speed Test ===" << std::endl;
        std::vector<uint32_t> ef_values = {10, 20, 50, 100, 200};
        
        for (uint32_t test_ef : ef_values) {
            start = std::chrono::high_resolution_clock::now();
            auto test_results = hnsw.search(queries[0], test_ef);
            end = std::chrono::high_resolution_clock::now();
            
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "ef=" << test_ef << ": " << test_results.size() 
                      << " results in " << duration.count() << " μs" << std::endl;
        }
        
        // Enable profiling to count distance calculations
        std::cout << "\n=== Distance Calculation Profiling ===" << std::endl;
        enableProfiling();
        resetProfilingCounter();
        
        hnsw.search(queries[0], 50);
        uint64_t dist_calcs = getCountDistCalc();
        std::cout << "Distance calculations for ef=50: " << dist_calcs << std::endl;
        
        resetProfilingCounter();
        hnsw.search(queries[0], 100);
        dist_calcs = getCountDistCalc();
        std::cout << "Distance calculations for ef=100: " << dist_calcs << std::endl;
        
        disableProfiling();
        
        // Test search accuracy (if you have ground truth)
        std::cout << "\n=== Search Results Analysis ===" << std::endl;
        auto detailed_results = hnsw.search(queries[0], 100);
        
        std::cout << "Distance distribution for top 20 results:" << std::endl;
        for (int i = 0; i < std::min(20, (int)detailed_results.size()); i++) {
            std::cout << "  " << i+1 << ": " << detailed_results[i].first << std::endl;
        }
        
        // Check if distances are properly sorted (should be ascending)
        bool is_sorted = std::is_sorted(detailed_results.begin(), detailed_results.end(),
            [](const search_result_t& a, const search_result_t& b) {
                return a.first < b.first;
            });
        
        std::cout << "Results properly sorted: " << (is_sorted ? "YES" : "NO") << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Failed to load index: " << e.what() << std::endl;
        std::cout << "Please run test_hnsw first to create an index." << std::endl;
        return 1;
    }
    
    std::cout << "\nSearch test completed!" << std::endl;
    return 0;
}