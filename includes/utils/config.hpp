#pragma once

#include <string>
#include <cstddef>

namespace Config
{
    // Enable verbose logging for debugging
    constexpr const bool VERBOSE = true;
`
    namespace Inference
    {
        // constexpr const char MODEL_PATH[] = "models/finetuned_sgn33-new-a-Apr6.xml";
        constexpr const char MODEL_PATH[] = "/home/tam/tam-workspace/Research/Optimized-DeepAligner-CPU/models/finetuned_sgn33-new-a-Apr6.xml";
        constexpr const size_t BATCH_SIZE = 100;
        constexpr const size_t MAX_LEN = 123;
        constexpr const size_t MODEL_OUT_SIZE = 128;
        constexpr const size_t NUM_THREADS = 8;         // Number of threads used in OpenVINO inference
        constexpr const size_t NUM_STREAMS = 4;         // OpenVino streams
        constexpr const size_t NUM_INFER_REQUESTS = 64; // Number of inference requests handled concurrently
    }

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
        // Batch during search, turn on for large datasets (Not use for now)
        constexpr const bool ENABLE_BATCH = false;
        constexpr const int BATCH_SIZE = 128;

        // Search parameters
        constexpr const int EF = 128;
        constexpr const int K = 128; // K <= EF

        // Multi-threaded search parameters
        constexpr const int NUM_THREADS = 128; // Number of threads for parallel search
    }

}