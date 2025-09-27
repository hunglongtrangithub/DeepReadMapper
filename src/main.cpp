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
        std::cerr << "Usage: " << argv[0] << " <index_prefix> <quer_seqs.fastq> <ref_seqs.fasta> [EF] [K] [output_dir] [use_dynamic]" << std::endl;
        std::cerr << "  - EF: Optional HNSW search parameter (default: " << Config::Search::EF << ")" << std::endl;
        std::cerr << "  - K: Optional number of nearest neighbors to return (default: " << Config::Search::K << ")" << std::endl;

        std::cerr << "  - output_dir: Optional output directory (default: current directory)" << std::endl;
        std::cerr << "  - use_dynamic: Optional flag to load reference sequences dynamically (1) or statically (0). Default: 0" << std::endl;
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

        size_t ref_len = std::get<size_t>(config["ref_len"]);
        size_t stride = std::get<size_t>(config["stride"]);

        const std::string query_seqs_file = argv[2];
        const std::string ref_seqs_file = argv[3];

        // Optional HNSW search parameters
        const int ef = (argc >= 5) ? std::stoi(argv[4]) : Config::Search::EF;
        const int k = (argc >= 6) ? std::stoi(argv[5]) : Config::Search::K;

        // Optional output file names with defaults
        const std::string output_dir = (argc >= 7) ? argv[6] : ".";
        const std::string sam_file = output_dir + "/results.sam";

        //* Suggest: Use dynamic when ref_len is large (e.g. 10,000) to save memory
        const bool use_dynamic = (argc >= 8) ? (std::stoi(argv[7]) != 0) : false;

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

        auto query_end_time = std::chrono::high_resolution_clock::now();
        auto query_duration = std::chrono::duration_cast<std::chrono::milliseconds>(query_end_time - start_time);

        // analyze_input(query_sequences);

        // Load reference sequences with stride=1 to get full sequences for post-processing

        std::string ref_genome = "";
        std::vector<std::string> ref_sequences = {};
        if (use_dynamic)
        {
            std::cout << "[MAIN] Using DYNAMIC fetching for reference sequences" << std::endl;
            ref_genome = extract_FASTA_sequence(ref_seqs_file);
        }
        else
        {
            std::cout << "[MAIN] Using STATIC fetching for reference sequences" << std::endl;
            ref_sequences = read_file(ref_seqs_file, ref_len, 1, true).first;

            std::cout << "[MAIN] Loaded " << ref_sequences.size() << " reference sequences from " << ref_seqs_file << std::endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[MAIN] Finished data loading" << std::endl;
        std::cout << "[MAIN] Query loading & formatting time: " << query_duration.count() << " ms" << std::endl;
        std::cout << "[MAIN] Reference loading & formatting time: " << (duration - query_duration).count() << " ms" << std::endl;
        std::cout << "[MAIN] Total Data loading time: " << duration.count() << " ms" << std::endl
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

        // Free-up search memory
        delete alg_hnsw;

        std::cout << "[MAIN] POST-PROCESSING STEP" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        // Declare variables outside the if-else blocks
        std::vector<std::string> final_seqs;
        std::vector<float> final_dists;

        //* L2 distance reranking
        int rerank_lim = Config::PostProcess::RERANK_LIM;
        if (use_dynamic)
        {
            std::tie(final_seqs, final_dists) = post_process_l2_dynamic(neighbors, distances, ref_genome, query_sequences, ref_len, stride, k, embeddings, vectorizer, rerank_lim);
        }
        else
        {
            std::tie(final_seqs, final_dists) = post_process_l2_static(neighbors, distances, ref_sequences, query_sequences, ref_len, stride, k, embeddings, vectorizer, rerank_lim);
        }

        //* Smith-Waterman reranking
        // if (use_dynamic)
        // {
        //     std::tie(final_seqs, final_scores) = post_process_sw_dynamic(neighbors, distances, ref_genome, query_sequences, ref_len, stride, k, rerank_lim);
        // }
        // else
        // {
        //     std::tie(final_seqs, final_scores) = post_process_sw_static(neighbors, distances, ref_sequences, query_sequences, ref_len, stride, k, rerank_lim);
        // }

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[MAIN] Post-processing completed" << std::endl;
        std::cout << "[MAIN] Post-processing time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // Print first 5 cands of first 3 queries for verification
        for (size_t i = 0; i < std::min(size_t(3), query_sequences.size()); ++i)
        {
            std::cout << "Query " << i << " - reranked candidates:" << std::endl;
            for (size_t j = 0; j < std::min(size_t(5), size_t(k)); ++j)
            {
                size_t idx = i * k + j;
                std::cout << "  Cand " << j << ": " << final_seqs[idx] << " (Distance: " << final_dists[idx] << ")" << std::endl;
            }
        }

        // Save results to disk
        //! This is deprecated
        // TODO: Replace from bin/npy output to SAM format
        // std::cout << "[MAIN] OUTPUT SAVING STEP" << std::endl;
        // start_time = std::chrono::high_resolution_clock::now();

        // bool use_npy = true;
        // std::string indices_file = output_dir + "/neighbors.npy";
        // std::string distances_file = output_dir + "/distances.npy";
        // save_results(neighbors, distances, indices_file, distances_file, k, use_npy);

        // end_time = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        // std::cout << "[MAIN] Output saving time: " << duration.count() << " ms" << std::endl;

        // auto master_end = std::chrono::high_resolution_clock::now();
        // auto master_duration = std::chrono::duration_cast<std::chrono::milliseconds>(master_end - master_start);
        // std::cout << "[MAIN] Total pipeline time: " << master_duration.count() << " ms" << std::endl
        //           << std::endl;

        std::cout << "=== Pipeline Completed Successfully! ===" << std::endl;
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