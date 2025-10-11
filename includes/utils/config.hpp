#pragma once

#include <string>
#include <cstddef>

namespace Config
{
    constexpr const bool VERBOSE = true; // Enable verbose logging for debug

    namespace Preprocess
    {
        constexpr const int NUM_THREADS = 128; // Number of threads for parallel preprocessing
        constexpr const int CHUNK_SIZE = 1000; // Number of sequences per chunk during preprocessing. 1 thread handles 1 chunk at a time.
    }

    namespace Inference
    {
        // constexpr const char MODEL_PATH[] = "models/finetuned_sgn33-new-a-Apr6.xml";
        constexpr const char MODEL_PATH[] = "/home/tam/tam-workspace/Research/Optimized-DeepAligner-CPU/models/finetuned_sgn33-new-a-Apr6.xml"; // Path to the model file
        constexpr const size_t BATCH_SIZE = 100;                                                                                                // Max num of sequences/batch. Actual batch may be smaller.
        constexpr const size_t MAX_LEN = 123;                                                                                                   // Max sequence length handled by the model. If shorter or longer, will be padded/truncated.
        constexpr const size_t MODEL_OUT_SIZE = 128;                                                                                            // Output vector size from the model
        constexpr const size_t NUM_THREADS = 128;                                                                                               // Num threads used in OpenVINO inference
        constexpr const size_t NUM_STREAMS = 128;                                                                                               // OpenVino streams (either using NUM_STREAMS or NUM_THREADS, not both)
        constexpr const size_t NUM_INFER_REQUESTS = 2048;                                                                                       // Max num of inference requests handled concurrently (1 request = 1 batch)
    }

    namespace Build
    {
        constexpr const int EFC = 128;    // Exploration factor for HNSW
        constexpr const int GPH_DEG = 64; // Graph degree (or M) in HNSW

        constexpr const int NUM_THREADS = 128; // Number of threads for parallel search

        constexpr const double SAMPLE_RATE = 0.1; // Fraction of data to use for training PQ codebooks

        //! Below are deprecated and only use for testing purpose
        constexpr const int DIM = 128;              // Vector dimension. Keep it const for testing.
        constexpr const int MAX_ELEMENTS = 1000000; // Max number of elements in the index, keep it const for testing.
    }

    namespace Search
    {
        constexpr const bool ENABLE_BATCH = false; // Enable batching during search
        constexpr const int BATCH_SIZE = 128;      // Number of queries per batch
        constexpr const int EF = 128;              // Search params. Higher EF -> more precise but slower
        constexpr const int K = 128;               // Top K results to return, K <= EF
        constexpr const int K_CLUSTERS = 5;        // [Sparse Index] Number of clusters (sparse candidates) to search for each query, K <= K_CLUSTERS * 2 * stride & K_CLUSTERS <= EF
        constexpr const int NUM_THREADS = 128;     // Number of threads for parallel search
    }

    namespace Postprocess
    {
        constexpr const int NUM_THREADS = 128;            // Number of threads for parallel post-processing and reranking
        constexpr const int CHUNK_SIZE = 10000000;        // Number of candidates per batch during L2 reranking (deprecated)
        constexpr const size_t QUERY_BATCH_SIZE = 5000; // Number of queries to process in a batch during post-processing
    }
}