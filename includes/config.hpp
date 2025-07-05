#pragma once

#include <cstddef> // for size_t

namespace Config
{
    // Enable verbose logging for debugging
    constexpr const bool VERBOSE = true;

    namespace Build
    {
        // Vector Dimension
        constexpr const int DIM = 128;
    
        // Exploration factor
        constexpr const int EFC = 128;

        // Graph degree GPH_DEG or M (in HNSW)
        constexpr const int GPH_DEG = 64;

        constexpr const int MAX_ELEMENTS = 1000000; // Maximum number of elements in the index, keep it const for testing. Later use the ref file length to have precise value.
    }

    namespace Search
    {
        // Batch during search, turn on for large datasets
        constexpr const bool ENABLE_BATCH = false;
        constexpr const int BATCH_SIZE = 128;

        // Search parameters
        constexpr const int EF = 100;
        constexpr const int K = 100; // K <= EF
    }
}
