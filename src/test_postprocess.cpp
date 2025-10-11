#include "cnpy.h"
#include "config.hpp"
#include "utils.hpp"
#include "post_processor.hpp"
#include "vectorize.hpp"
#include <filesystem>

int main()
{
    try
    {
        auto master_start = std::chrono::high_resolution_clock::now();
        std::cout << "=== DeepAligner Post-Processing Test ===" << std::endl
                  << std::endl;

        // Read from command line arguments
        const std::string indices_file = "/home/tam/tam-store3/temp_output/ecoli_150_sparse/indices.npy";
        const std::string distances_file = "/home/tam/tam-store3/temp_output/ecoli_150_sparse/distances.npy";
        const std::string query_seqs_file = "/mnt/7T-ssdD/species-datasets/ecoli/ecoli_150/ecoli_150.fastq";

        const std::string ref_seqs_file = "/mnt/7T-ssdD/species-datasets/fasta_source/ecoli.fna";

        // Optional parameters
        const int k = 20;
        const int k_clusters = 15;
        const bool use_dynamic = false;

        // Config inference parameters (needed for reranking)
        const std::string model_path = Config::Inference::MODEL_PATH;
        const size_t batch_size = Config::Inference::BATCH_SIZE;
        const size_t max_len = Config::Inference::MAX_LEN;
        const size_t model_out_size = Config::Inference::MODEL_OUT_SIZE;

        // Load index config (fixed for debug)
        std::string index_dir = "/home/tam/tam-store/sparse_index/ecoli_150";
        std::string config_file = index_dir + "/config.txt";

        size_t ref_len, stride;
        if (std::filesystem::exists(config_file))
        {
            std::cout << "[TEST] Loading config from: " << config_file << std::endl;
            std::unordered_map<std::string, ConfigValue> config = load_config(config_file);
            ref_len = std::get<size_t>(config["ref_len"]);
            stride = std::get<size_t>(config["stride"]);
        }
        else
        {
            std::cerr << "[TEST] Warning: Config file not found. Terminate." << std::endl;
            return 1;
        }

        std::cout << "[TEST] TEST CONFIG:" << std::endl;
        std::cout << "[TEST] Indices file: " << indices_file << std::endl;
        std::cout << "[TEST] Distances file: " << distances_file << std::endl;
        std::cout << "[TEST] Query file: " << query_seqs_file << std::endl;
        std::cout << "[TEST] Reference file: " << ref_seqs_file << std::endl;
        std::cout << "[TEST] K: " << k << std::endl;
        std::cout << "[TEST] Ref length: " << ref_len << std::endl;
        std::cout << "[TEST] Stride: " << stride << std::endl;
        std::cout << "[TEST] Use dynamic: " << (use_dynamic ? "true" : "false") << std::endl
                  << std::endl;

        // Load indices and distances from .npy files
        std::cout << "[TEST] LOADING NPY FILES" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        cnpy::NpyArray indices_arr = cnpy::npy_load(indices_file);
        cnpy::NpyArray distances_arr = cnpy::npy_load(distances_file);

        // Verify shapes
        if (indices_arr.shape.size() != 2 || distances_arr.shape.size() != 2)
        {
            throw std::runtime_error("Invalid .npy file shape. Expected 2D arrays.");
        }

        size_t num_queries = indices_arr.shape[0];
        size_t k_loaded = indices_arr.shape[1];

        if (distances_arr.shape[0] != num_queries || distances_arr.shape[1] != k_loaded)
        {
            throw std::runtime_error("Indices and distances arrays have mismatched shapes.");
        }

        std::cout << "[TEST] Loaded arrays with shape: [" << num_queries << ", " << k_loaded << "]" << std::endl;

        // Convert to std::vector format
        std::vector<std::vector<size_t>> neighbors(num_queries);
        std::vector<std::vector<float>> distances(num_queries);

        size_t *indices_data = indices_arr.data<size_t>();
        float *distances_data = distances_arr.data<float>();

        for (size_t i = 0; i < num_queries; ++i)
        {
            neighbors[i].resize(k_loaded);
            distances[i].resize(k_loaded);

            for (size_t j = 0; j < k_loaded; ++j)
            {
                neighbors[i][j] = indices_data[i * k_loaded + j];
                distances[i][j] = distances_data[i * k_loaded + j];
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] NPY loading time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // Read query sequences
        std::cout << "[TEST] LOADING QUERY SEQUENCES" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        auto [query_sequences, __] = read_file(query_seqs_file);

        if (query_sequences.empty())
        {
            std::cerr << "No query sequences found in input file!" << std::endl;
            return 1;
        }

        if (query_sequences.size() != num_queries)
        {
            throw std::runtime_error("Number of query sequences (" + std::to_string(query_sequences.size()) +
                                     ") doesn't match number of queries in npy file (" + std::to_string(num_queries) + ")");
        }

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Query loading time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // Load reference sequences
        std::cout << "[TEST] LOADING REFERENCE SEQUENCES" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::string> ref_sequences = {};
        std::cout << "[TEST] Using STATIC fetching for reference sequences" << std::endl;
        ref_sequences = read_file(ref_seqs_file, ref_len, 1, true).first;
        std::cout << "[TEST] Loaded " << ref_sequences.size() << " reference sequences" << std::endl;

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Reference loading time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // Replace everything from line 153 onwards with this:

        // ============================================================
        // SIMPLIFIED: Call post_process_l2_static directly
        // ============================================================
        std::cout << "[TEST] CALLING post_process_l2_static DIRECTLY" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        // Initialize vectorizer
        Vectorizer vectorizer(model_path, batch_size, max_len, model_out_size);

        // Vectorize queries
        std::vector<std::vector<float>> query_embeddings = vectorizer.vectorize(query_sequences);

        std::cout << "[TEST] Query embeddings: " << query_embeddings.size() << " x " << query_embeddings[0].size() << std::endl;

        // Call post_process_l2_static (it handles everything internally!)
        auto [final_seqs, final_dists, final_ids] = post_process_l2_static(
            neighbors,
            distances,
            ref_sequences,
            query_sequences,
            ref_len,
            stride,
            k,
            query_embeddings,
            vectorizer,
            k_clusters);

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] post_process_l2_static completed in " << duration.count() << " ms" << std::endl
                  << std::endl;

        // ============================================================
        // VERIFY FINAL RESULTS
        // ============================================================
        std::cout << "[TEST] VERIFYING FINAL RESULTS" << std::endl;

        // Print first few results for verification
        std::cout << "[TEST] Sample results (first 3 queries, first 5 candidates each):" << std::endl;
        for (size_t i = 0; i < std::min(size_t(3), num_queries); ++i)
        {
            std::cout << "Query " << i << " - reranked candidates:" << std::endl;
            for (size_t j = 0; j < std::min(size_t(5), size_t(k)); ++j)
            {
                size_t idx = i * k + j;
                if (idx < final_seqs.size())
                {
                    std::cout << "  Cand " << j << ": ID=" << final_ids[idx]
                              << ", L2_Distance=" << final_dists[idx]
                              << ", Seq=" << final_seqs[idx].substr(0, 50) << std::endl;
                }
            }
        }
        std::cout << std::endl;

        std::cout << "[TEST] Final results shape:" << std::endl;
        std::cout << "  Sequences: " << final_seqs.size() << std::endl;
        std::cout << "  Distances: " << final_dists.size() << std::endl;
        std::cout << "  IDs: " << final_ids.size() << std::endl
                  << std::endl;

        auto master_end = std::chrono::high_resolution_clock::now();
        auto master_duration = std::chrono::duration_cast<std::chrono::milliseconds>(master_end - master_start);
        std::cout << "[TEST] Total test time: " << master_duration.count() << " ms" << std::endl
                  << std::endl;

        std::cout << "=== Test Completed Successfully! ===" << std::endl;
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