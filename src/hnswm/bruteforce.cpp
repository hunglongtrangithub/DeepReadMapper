#include "bruteforce.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <assert.h>
#include <immintrin.h>
#include <queue>
#include <omp.h>
#include <mutex>
#include <functional>

BruteForce::BruteForce(uint32_t dim, uint64_t maxNumNodes)
{
    vectors_.reserve(maxNumNodes);
}

void BruteForce::buildIndex(const std::vector<std::vector<float>> &vectors)
{
    this->vectors_.insert(
        this->vectors_.end(),
        vectors.begin(), vectors.end());
}

std::vector<search_result_t> BruteForce::search(const std::vector<float> &query, uint32_t k)
{
    std::priority_queue<search_result_t, std::vector<search_result_t>, std::greater<search_result_t>> max_heap;
    std::mutex heap_mutex; // Mutex to protect heap access

#pragma omp parallel
    {
        // Local priority queue for each thread
        std::priority_queue<search_result_t, std::vector<search_result_t>, std::greater<search_result_t>> local_max_heap;

#pragma omp for nowait // Distribute loop iterations across threads
        for (size_t i = 0; i < vectors_.size(); ++i)
        {
            float distance = squaredEuclideanDistance(query, vectors_[i]);

            // Access to the local max heap does not need to be synchronized
            if (local_max_heap.size() < k)
            {
                local_max_heap.push({distance, static_cast<nodeID_t>(i)});
            }
            else if (distance < local_max_heap.top().first)
            {
                local_max_heap.pop();
                local_max_heap.push({distance, static_cast<nodeID_t>(i)});
            }
        }

// Merge local heaps into the global heap
#pragma omp critical
        {
            while (!local_max_heap.empty())
            {
                auto top = local_max_heap.top();
                local_max_heap.pop();
                if (max_heap.size() < k)
                {
                    max_heap.push(top);
                }
                else if (top.first < max_heap.top().first)
                {
                    max_heap.pop();
                    max_heap.push(top);
                }
            }
        }
    }

    // Extract the elements from the global heap and store them in a vector
    std::vector<search_result_t> k_nearest;
    while (!max_heap.empty())
    {
        k_nearest.push_back(max_heap.top());
        max_heap.pop();
    }

    // The elements are in reverse order since it's a max-heap, so reverse them
    std::reverse(k_nearest.begin(), k_nearest.end());

    return k_nearest;
}

void BruteForce::save(const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open())
    {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    cereal::BinaryOutputArchive archive(ofs);
    archive(*this);
}

BruteForce loadBruteForce(const std::string &filename)
{
    BruteForce bf(0); // Temporary object
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    cereal::BinaryInputArchive archive(ifs);
    archive(bf);
    return bf;
}