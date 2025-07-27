#include "cnpy.h"
#include "config.hpp"
#include "utils.hpp"
#include "vectorize.hpp"
#include "hnswlib_dir/search.hpp"
#include <filesystem>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

/*
Currently In Progress:

This code implements a pipelined approach to current workflow to seek potential speed up.
*/

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 5)
    {
        std::cerr << "Usage: " << argv[0] << " <search.index> <query_seq.txt> [indices_output.npy] [distances_output.npy]" << std::endl;
        std::cerr << "  - indices_output.npy: Optional indices output file (default: indices.npy)" << std::endl;
        std::cerr << "  - distances_output.npy: Optional distances output file (default: distances.npy)" << std::endl;
        return 1;
    }

    try
    {
        auto master_start = std::chrono::high_resolution_clock::now();
        std::cout << "=== DeepAligner CPU Pipeline ===" << std::endl
                  << std::endl;

        // Read from command line arguments
        const std::string index_file = argv[1];
        const std::string sequences_file = argv[2];
        const std::string indices_file = (argc >= 4) ? argv[3] : "indices.npy";
        const std::string distances_file = (argc >= 5) ? argv[4] : "distances.npy";

        // Config parameters
        const std::string model_path = Config::Inference::MODEL_PATH;
        const size_t batch_size = Config::Inference::BATCH_SIZE;
        const size_t max_len = Config::Inference::MAX_LEN;
        const size_t model_out_size = Config::Inference::MODEL_OUT_SIZE;
        const int dim = Config::Build::DIM;
        const int ef = Config::Search::EF;

        std::cout << "[MAIN] PIPELINE CONFIG:" << std::endl;
        std::cout << "[MAIN] Input file: " << sequences_file << std::endl;
        std::cout << "[MAIN] Model path: " << model_path << std::endl;
        std::cout << "[MAIN] Batch size: " << batch_size << std::endl;
        std::cout << "[MAIN] Max sequence length: " << max_len << std::endl;
        std::cout << "[MAIN] Model output size: " << model_out_size << std::endl;

        // Load data
        std::cout << "[MAIN] DATA LOADING STEP" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<std::string> sequences = read_file(sequences_file);
        if (sequences.empty())
        {
            std::cerr << "No sequences found in input file!" << std::endl;
            return 1;
        }
        analyze_input(sequences);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[MAIN] Data loaded time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // Load HNSW index
        std::cout << "[MAIN] HNSW INDEX LOADING STEP" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        if (!std::filesystem::exists(index_file))
        {
            throw std::runtime_error("Index file does not exist: " + index_file);
        }
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_file);
        alg_hnsw->setEf(ef);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[MAIN] Index loaded time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // Initialize vectorizer
        Vectorizer vectorizer(model_path, batch_size, max_len, model_out_size);

        // Pipeline setup
        std::cout << "[MAIN] STARTING PIPELINED INFERENCE + SEARCH" << std::endl;
        const size_t chunk_size = batch_size * Config::Inference::NUM_INFER_REQUESTS / 2;
        const size_t total_sequences = sequences.size();

        // Results storage
        std::vector<std::vector<hnswlib::labeltype>> neighbors(total_sequences);
        std::vector<std::vector<float>> distances(total_sequences);

        // Pipeline communication
        struct EmbeddingChunk
        {
            std::vector<std::vector<float>> embeddings;
            size_t start_idx;
            bool is_last;
        };

        std::queue<EmbeddingChunk> embedding_queue;
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        bool inference_done = false;

        auto pipeline_start = std::chrono::high_resolution_clock::now();

        // Producer thread: Inference
        std::thread inference_thread([&]()
                                     {
            for (size_t chunk_start = 0; chunk_start < total_sequences; chunk_start += chunk_size) {
                size_t chunk_end = std::min(chunk_start + chunk_size, total_sequences);
                
                // Extract chunk sequences
                std::vector<std::string> chunk_sequences(
                    sequences.begin() + chunk_start,
                    sequences.begin() + chunk_end
                );

                // Run inference on chunk
                std::vector<std::vector<float>> chunk_embeddings = vectorizer.vectorize(chunk_sequences);

                // Add to queue
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    embedding_queue.push({
                        std::move(chunk_embeddings), 
                        chunk_start, 
                        chunk_end >= total_sequences
                    });
                }
                queue_cv.notify_one();
            }

            // Signal inference completion
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                inference_done = true;
            }
            queue_cv.notify_all(); });

        // Consumer thread: Search
        std::thread search_thread([&]()
                                  {
            while (true) {
                EmbeddingChunk chunk;
                
                // Wait for next chunk
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    queue_cv.wait(lock, [&] { return !embedding_queue.empty() || inference_done; });
                    
                    if (embedding_queue.empty() && inference_done) {
                        break; // No more work
                    }
                    
                    if (!embedding_queue.empty()) {
                        chunk = std::move(embedding_queue.front());
                        embedding_queue.pop();
                    } else {
                        continue; // Spurious wakeup
                    }
                }

                // Run search on chunk
                auto [chunk_neighbors, chunk_distances] = search(alg_hnsw, chunk.embeddings);

                // Store results
                for (size_t i = 0; i < chunk_neighbors.size(); ++i) {
                    neighbors[chunk.start_idx + i] = std::move(chunk_neighbors[i]);
                    distances[chunk.start_idx + i] = std::move(chunk_distances[i]);
                }
            } });

        // Wait for both threads to complete
        inference_thread.join();
        search_thread.join();

        auto pipeline_end = std::chrono::high_resolution_clock::now();
        auto pipeline_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pipeline_end - pipeline_start);
        std::cout << "[MAIN] Pipelined inference + search time: " << pipeline_duration.count() << " ms" << std::endl
                  << std::endl;

        // Save results
        std::cout << "[MAIN] OUTPUT SAVING STEP" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        size_t n_rows = neighbors.size();
        size_t k = Config::Search::K;

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

        cnpy::npy_save(indices_file, host_indices.data(), {static_cast<unsigned long>(n_rows), static_cast<unsigned long>(k)});
        cnpy::npy_save(distances_file, host_distances.data(), {static_cast<unsigned long>(n_rows), static_cast<unsigned long>(k)});

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "[MAIN] Results saved to " << indices_file << " and " << distances_file << std::endl;
        std::cout << "[MAIN] Output saving time: " << duration.count() << " ms" << std::endl;

        auto master_end = std::chrono::high_resolution_clock::now();
        auto master_duration = std::chrono::duration_cast<std::chrono::milliseconds>(master_end - master_start);
        std::cout << "[MAIN] Total pipeline time: " << master_duration.count() << " ms" << std::endl
                  << std::endl;

        std::cout << "=== Pipeline Completed Successfully! ===" << std::endl;

        delete alg_hnsw;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown error occurred!" << std::endl;
        return 1;
    }

    return 0;
}