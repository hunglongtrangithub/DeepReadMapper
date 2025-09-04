#include "cnpy.h"
#include "config.hpp"
#include "utils.hpp"
#include "post_processor.hpp"
#include "vectorize.hpp"
// #include "hnswlib_dir/search.hpp"
#include "hnswpq/search.hpp"
#include <filesystem>

int main(int argc, char *argv[])
{
    if (argc < 4 || argc > 8)
    {
        std::cerr << "Usage: " << argv[0] << " <index_prefix> <quer_seqs.fastq> <ref_seqs.fasta> [EF] [K] [output_dir] [use_npy]" << std::endl;
        std::cerr << "  - EF: Optional HNSW search parameter (default: " << Config::Search::EF << ")" << std::endl;
        std::cerr << "  - K: Optional number of nearest neighbors to return (default: " << Config::Search::K << ")" << std::endl;

        std::cerr << "  - output_dir: Optional output directory (default: current directory)" << std::endl;
        std::cerr << "  - use_npy: Optional flag to save results in .npy format (default: false)" << std::endl;
        return 1;
    }

    try
    {
        auto master_start = std::chrono::high_resolution_clock::now();
        std::cout << "=== DeepAligner CPU Pipeline ===" << std::endl
                  << std::endl;

        // Read from command line arguments
        const std::string index_prefix = argv[1];

        // Craft index file name and folder structure
        std::string basename = std::filesystem::path(index_prefix).filename().string();

        const std::string index_file = index_prefix + "/" + basename + ".index";
        const std::string config_file = index_prefix + "/" + "config.txt";

        // Load index config
        if (!std::filesystem::exists(config_file))
        {
            throw std::runtime_error("Config file does not exist: " + config_file);
        }
        std::unordered_map<std::string, ConfigValue> config = load_config(config_file);

        const std::string query_seqs_file = argv[2];
        const std::string ref_seqs_file = argv[3];

        // Optional HNSW search parameters
        const int ef = (argc >= 5) ? std::stoi(argv[4]) : Config::Search::EF;
        const int k = (argc >= 6) ? std::stoi(argv[5]) : Config::Search::K;

        // Optional output file names with defaults
        const std::string output_dir = (argc >= 7) ? argv[6] : ".";

        const bool use_npy = (argc >= 8) ? std::string(argv[7]) == "true" : false;

        // Craft full output paths
        const std::string indices_file = output_dir + (use_npy ? "/indices.npy" : "/indices.bin");
        const std::string distances_file = output_dir + (use_npy ? "/distances.npy" : "/distances.bin");

        // Config inference parameters
        const std::string model_path = Config::Inference::MODEL_PATH;
        const size_t batch_size = Config::Inference::BATCH_SIZE;
        const size_t max_len = Config::Inference::MAX_LEN;
        const size_t model_out_size = Config::Inference::MODEL_OUT_SIZE;

        std::cout << "[MAIN] PIPELINE CONFIG:" << std::endl;
        std::cout << "[MAIN] Input file: " << query_seqs_file << std::endl;
        std::cout << "[MAIN] Reference file: " << ref_seqs_file << std::endl;
        std::cout << "[MAIN] Model path: " << model_path << std::endl;
        std::cout << "[MAIN] Batch size: " << batch_size << std::endl;
        std::cout << "[MAIN] Max sequence length: " << max_len << std::endl;
        std::cout << "[MAIN] Model output size: " << model_out_size << std::endl;
        std::cout << "[MAIN] Indices output: " << indices_file << std::endl;
        std::cout << "[MAIN] Distances output: " << distances_file << std::endl;

        // Read sequences from file
        std::cout << "[MAIN] DATA LOADING STEP" << std::endl;
        std::cout << "[MAIN] Reading sequences from Disk" << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        auto [query_sequences, __] = read_file(query_seqs_file);

        if (query_sequences.empty())
        {
            std::cerr << "No query_sequences found in input file!" << std::endl;
            return 1;
        }

        // analyze_input(query_sequences);

        auto [ref_sequences, _] = read_file(ref_seqs_file, std::get<size_t>(config["ref_len"]), std::get<size_t>(config["stride"]));

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[MAIN] Finished data loading" << std::endl;
        std::cout << "[MAIN] Data loaded time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // Load search index
        const int dim = Config::Build::DIM;

        std::cout << "[MAIN] HNSW INDEX LOADING STEP" << std::endl;
        std::cout << "[MAIN] Search Index Config:" << std::endl;
        std::cout << "[MAIN] Index file: " << index_file << std::endl;
        std::cout << "[MAIN] Dimension: " << dim << std::endl;
        std::cout << "[MAIN] EF: " << ef << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        if (!std::filesystem::exists(index_file))
        {
            throw std::runtime_error("Index file does not exist: " + index_file);
        }
        //* Load Original HNSW index
        // hnswlib::L2Space space(dim);
        // hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_file);

        //* Load HNSWPQ index
        faiss::Index *loaded_index = faiss::read_index(index_file.c_str());
        faiss::IndexHNSWPQ *alg_hnsw = dynamic_cast<faiss::IndexHNSWPQ *>(loaded_index);

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "[MAIN] Finished loading index" << std::endl;
        std::cout << "[MAIN] Index loaded time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // Initialize vectorizer
        std::cout << "[MAIN] INFERENCE STEP" << std::endl;
        Vectorizer vectorizer(model_path, batch_size, max_len, model_out_size);

        // Run vectorization
        start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<float>> embeddings = vectorizer.vectorize(query_sequences);

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Print results
        std::cout << "[MAIN] Inference completed" << std::endl;
        std::cout << "[MAIN] Inference time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        std::cout << "[MAIN] HNSW SEARCH STEP" << std::endl;
        std::cout << "[MAIN] Searching for nearest neighbors..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        //* Original HNSW search
        // auto [neighbors, distances] = search(alg_hnsw, embeddings);

        //* HNSWPQ search
        auto [neighbors, distances] = faiss_search(alg_hnsw, embeddings, k, ef);

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[MAIN] HNSW Search completed" << std::endl;
        std::cout << "[MAIN] Search time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        std::cout << "[MAIN] POST-PROCESSING STEP" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        size_t ref_len = std::get<size_t>(config["ref_len"]);
        size_t stride = std::get<size_t>(config["stride"]);


        auto [final_seqs, sw_scores] = post_process(neighbors, distances, ref_sequences, query_sequences, ref_len, stride, k);

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[MAIN] Post-processing completed" << std::endl;
        std::cout << "[MAIN] Post-processing time: " << duration.count() << " ms" << std::endl << std::endl;

        // Print first 10 cands of first 5 queries for verification
        for (size_t i = 0; i < std::min(size_t(5), query_sequences.size()); ++i)
        {
            std::cout << "Query " << i << " - reranked candidates:" << std::endl;
            for (size_t j = 0; j < std::min(size_t(10), size_t(k)); ++j)
            {
                size_t idx = i * k + j;
                std::cout << "  Cand " << j << ": " << final_seqs[idx] << " (SW-score: " << sw_scores[idx] << ")" << std::endl;
            }
        }

        // Save results to disk
        // std::cout << "[MAIN] OUTPUT SAVING STEP" << std::endl;
        // start_time = std::chrono::high_resolution_clock::now();

        // save_results(neighbors, distances, indices_file, distances_file, k, use_npy);

        // end_time = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // std::cout << "[MAIN] Results saved to " << indices_file << " and " << distances_file << std::endl;
        // std::cout << "[MAIN] Output saving time: " << duration.count() << " ms" << std::endl;

        auto master_end = std::chrono::high_resolution_clock::now();
        auto master_duration = std::chrono::duration_cast<std::chrono::milliseconds>(master_end - master_start);
        std::cout << "[MAIN] Total pipeline time: " << master_duration.count() << " ms" << std::endl
                  << std::endl;

        std::cout << "=== Pipeline Completed Successfully! ===" << std::endl;

        // Clean up
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