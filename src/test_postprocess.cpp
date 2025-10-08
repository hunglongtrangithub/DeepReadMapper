#include "cnpy.h"
#include "config.hpp"
#include "utils.hpp"
#include "post_processor.hpp"
#include "vectorize.hpp"
#include <filesystem>

int main(int argc, char *argv[])
{
    // Run with ./test_postprocess /home/tam/tam-store/hnswpq_results/ecoli_10000.hifi/neighbors.npy /home/tam/tam-store/hnswpq_results/ecoli_10000.hifi/distances.npy /home/tam/tam-store/data/long_fastq/ecoli/hifi/10000/ecoli_10000.hifi.fastq /home/tam/tam-store/data/fasta/ecoli.fna 128 0


    if (argc < 5 || argc > 7)
    {
        std::cerr << "Usage: " << argv[0] << " <indices.npy> <distances.npy> <quer_seqs.fastq> <ref_seqs.fasta> [K] [use_dynamic]" << std::endl;
        std::cerr << "  - indices.npy: Path to saved neighbor indices from HNSW search" << std::endl;
        std::cerr << "  - distances.npy: Path to saved distances from HNSW search" << std::endl;
        std::cerr << "  - quer_seqs.fastq: Query sequences file" << std::endl;
        std::cerr << "  - ref_seqs.fasta: Reference sequences file" << std::endl;
        std::cerr << "  - K: Optional number of nearest neighbors to return (default: " << Config::Search::K << ")" << std::endl;
        std::cerr << "  - use_dynamic: Optional flag to load reference sequences dynamically (1) or statically (0). Default: 0" << std::endl;
        return 1;
    }

    try
    {
        auto master_start = std::chrono::high_resolution_clock::now();
        std::cout << "=== DeepAligner Post-Processing Test ===" << std::endl
                  << std::endl;

        // Read from command line arguments
        const std::string indices_file = argv[1];
        const std::string distances_file = argv[2];
        const std::string query_seqs_file = argv[3];
        const std::string ref_seqs_file = argv[4];

        // Optional parameters
        const int k = (argc >= 6) ? std::stoi(argv[5]) : Config::Search::K;
        const bool use_dynamic = (argc >= 7) ? (std::stoi(argv[6]) != 0) : false;

        // Config inference parameters (needed for reranking)
        const std::string model_path = Config::Inference::MODEL_PATH;
        const size_t batch_size = Config::Inference::BATCH_SIZE;
        const size_t max_len = Config::Inference::MAX_LEN;
        const size_t model_out_size = Config::Inference::MODEL_OUT_SIZE;

        // Load index config (fixed for debug)
        std::string index_dir = "/home/tam/tam-store/hnswpq_index/ecoli_10000";
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

        std::string ref_genome = "";
        std::vector<std::string> ref_sequences = {};

        if (use_dynamic)
        {
            std::cout << "[TEST] Using DYNAMIC fetching for reference sequences" << std::endl;
            ref_genome = extract_FASTA_sequence(ref_seqs_file);
        }
        else
        {
            std::cout << "[TEST] Using STATIC fetching for reference sequences" << std::endl;
            ref_sequences = read_file(ref_seqs_file, ref_len, 1, true).first;
            std::cout << "[TEST] Loaded " << ref_sequences.size() << " reference sequences" << std::endl;
        }

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Reference loading time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // Initialize vectorizer for reranking
        std::cout << "[TEST] INITIALIZING VECTORIZER" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        Vectorizer vectorizer(model_path, batch_size, max_len, model_out_size);

        // Generate query embeddings for reranking
        std::vector<std::vector<float>> embeddings = vectorizer.vectorize(query_sequences);

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Vectorization time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // Run post-processing
        std::cout << "[TEST] POST-PROCESSING STEP" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::string> final_seqs;
        std::vector<float> final_dists;
        std::vector<size_t> final_ids;

        int rerank_lim = Config::Postprocess::RERANK_LIM;

        if (use_dynamic)
        {
            std::tie(final_seqs, final_dists, final_ids) = post_process_l2_dynamic(
                neighbors, distances, ref_genome, query_sequences,
                ref_len, stride, k, embeddings, vectorizer, rerank_lim);
        }
        else
        {
            std::tie(final_seqs, final_dists, final_ids) = post_process_l2_static(
                neighbors, distances, ref_sequences, query_sequences,
                ref_len, stride, k, embeddings, vectorizer, rerank_lim);
        }

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Post-processing time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // Print first few results for verification
        std::cout << "[TEST] SAMPLE RESULTS (first 3 queries, first 5 candidates each):" << std::endl;
        for (size_t i = 0; i < std::min(size_t(3), query_sequences.size()); ++i)
        {
            std::cout << "Query " << i << " - reranked candidates:" << std::endl;
            for (size_t j = 0; j < std::min(size_t(5), size_t(k)); ++j)
            {
                size_t idx = i * k + j;
                if (idx < final_seqs.size())
                {
                    std::cout << "  Cand " << j << ": ID=" << final_ids[idx]
                              << ", Distance=" << final_dists[idx]
                              << ", Seq=" << final_seqs[idx].substr(0, 50) << "..." << std::endl;
                }
            }
        }
        std::cout << std::endl;

        // Print shape of post-processed results
        std::cout << "[TEST] Post-processed results shape: [" << final_seqs.size() << ", " << (final_seqs.empty() ? 0 : final_seqs[0].size()) << "]" << std::endl
                  << std::endl;

        std::cout << "[TEST] Distance shape: [" << final_dists.size() << ", " << (final_dists.empty() ? 0 : 1) << "]" << std::endl;
        std::cout << "[TEST] IDs shape: [" << final_ids.size() << ", " << (final_ids.empty() ? 0 : 1) << "]" << std::endl
                  << std::endl;

        // Save results to SAM format
        std::cout << "[TEST] SAVING RESULTS TO SAM" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        std::string sam_file = "./results.sam";
        write_sam(final_seqs, final_dists, query_sequences, final_ids, "ref", ref_len, k, sam_file);

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] SAM output time: " << duration.count() << " ms" << std::endl;
        std::cout << "[TEST] Results saved to: " << sam_file << std::endl
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