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
    if (argc < 4 || argc > 10)
    {
        std::cerr << "Usage: " << argv[0] << " <index_prefix> <query_seqs.fastq> <ref_seqs.fasta> [EF] [K] [K_clusters] [output_dir] [use_dynamic] [use_streaming]" << std::endl;
        std::cerr << "  - query input: Can be FASTQ/FASTA/TXT file or pre-computed embeddings in .npy format" << std::endl;
        std::cerr << "  - EF: Optional HNSW search parameter (default: " << Config::Search::EF << ")" << std::endl;
        std::cerr << "  - K: Optional number of nearest neighbors to return (default: " << Config::Search::K << ")" << std::endl;
        std::cerr << "  - output_dir: Optional output directory (default: current directory)" << std::endl;
        std::cerr << "  - use_dynamic: Optional flag to load reference sequences dynamically (1) or statically (0). Default: 0" << std::endl;
        std::cerr << "  - use_streaming: Optional flag to use streaming output to SAM file (1) or normal output (0). Default: 0" << std::endl;
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

        const std::string query_input_file = argv[2];
        const std::string ref_seqs_file = argv[3];

        // Optional HNSW search parameters
        const int ef = (argc >= 5) ? std::stoi(argv[4]) : Config::Search::EF;
        const int k = (argc >= 6) ? std::stoi(argv[5]) : Config::Search::K;
        int k_clusters = Config::Search::K_CLUSTERS;
        if (stride == 1)
        {
            k_clusters = k;
        }
        else if (argc >= 7)
        {
            k_clusters = std::stoi(argv[6]);
        };

        // Optional output file names with defaults
        const std::string output_dir = (argc >= 8) ? argv[7] : ".";
        const std::string sam_file = output_dir + "/results.sam";

        //* Suggest: Use dynamic when ref_len is large (e.g. 10,000) to save memory
        const bool use_dynamic = (argc >= 9) ? (std::stoi(argv[8]) != 0) : false;
        const bool use_streaming = (argc >= 10) ? (std::stoi(argv[9]) != 0) : false;

        // Config inference parameters
        const std::string model_path = Config::Inference::MODEL_PATH;
        const size_t batch_size = Config::Inference::BATCH_SIZE;
        const size_t max_len = Config::Inference::MAX_LEN;
        const size_t model_out_size = Config::Inference::MODEL_OUT_SIZE;

        std::cout << "[MAIN] PIPELINE CONFIG:" << std::endl;
        std::cout << "[MAIN] Input file: " << query_input_file << std::endl;
        std::cout << "[MAIN] Reference file: " << ref_seqs_file << std::endl;
        std::cout << "[MAIN] Model path: " << model_path << std::endl;
        std::cout << "[MAIN] Batch size: " << batch_size << std::endl;
        std::cout << "[MAIN] Max sequence length: " << max_len << std::endl;
        std::cout << "[MAIN] Model output size: " << model_out_size << std::endl;

        // Main Pipeline = FASTQ + Vectorize + HNSW Search + Post-process + Output
        int main_pipeline_time = 0; // ms

        // Detect input file type
        std::string file_ext = std::filesystem::path(query_input_file).extension().string();
        bool is_precomputed_embeddings = (file_ext == ".npy");

        // Declare variables that will be used throughout the pipeline
        std::vector<std::vector<float>> embeddings;
        std::vector<std::string> query_sequences;
        std::vector<std::string> query_ids;
        std::chrono::milliseconds query_duration(0);

        if (is_precomputed_embeddings)
        {
            // Load pre-computed embeddings directly from .npy file
            std::cout << "[MAIN] DATA LOADING STEP" << std::endl;
            std::cout << "[MAIN] Detected pre-computed embeddings (.npy format)" << std::endl;
            std::cout << "[MAIN] Loading embeddings from: " << query_input_file << std::endl;

            auto start_time = std::chrono::high_resolution_clock::now();

            cnpy::NpyArray arr = cnpy::npy_load(query_input_file);

            if (arr.shape.size() != 2)
            {
                throw std::runtime_error("Error: Expected 2D array in .npy file");
            }

            size_t num_queries = arr.shape[0];
            size_t embedding_dim = arr.shape[1];

            std::cout << "[MAIN] Loaded " << num_queries << " embeddings of dimension " << embedding_dim << std::endl;

            // Convert to vector<vector<float>>
            float *data = arr.data<float>();
            embeddings.resize(num_queries);
            for (size_t i = 0; i < num_queries; ++i)
            {
                embeddings[i].resize(embedding_dim);
                for (size_t j = 0; j < embedding_dim; ++j)
                {
                    embeddings[i][j] = data[i * embedding_dim + j];
                }
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            main_pipeline_time += duration.count();

            std::cout << "[MAIN] Finished data loading" << std::endl;
            std::cout << "[MAIN] Embeddings loading time: " << duration.count() << " ms" << std::endl;
            std::cout << "[MAIN] Skipping inference step (using pre-computed embeddings)" << std::endl
                      << std::endl;

            // Note: query_sequences and query_ids remain empty - post-processing must handle this
            std::cout << "[MAIN] WARNING: No sequence data available for post-processing (only embeddings)" << std::endl;
            std::cout << "[MAIN] SAM output will not be generated without original sequences" << std::endl
                      << std::endl;
        }
        else
        {
            // Load sequences from FASTQ/FASTA/TXT and perform inference
            std::cout << "[MAIN] DATA LOADING STEP" << std::endl;
            std::cout << "[MAIN] Reading sequences from Disk" << std::endl;

            auto start_time = std::chrono::high_resolution_clock::now();

            std::tie(query_sequences, query_ids) = read_file(query_input_file);

            if (query_sequences.empty())
            {
                std::cerr << "No query_sequences found in input file!" << std::endl;
                return 1;
            }

            auto query_end_time = std::chrono::high_resolution_clock::now();
            query_duration = std::chrono::duration_cast<std::chrono::milliseconds>(query_end_time - start_time);

            main_pipeline_time += query_duration.count();

            std::cout << "[MAIN] Query loading & formatting time: " << query_duration.count() << " ms" << std::endl;
        }

        // Load reference sequences with stride=1 to get full sequences for post-processing
        std::string ref_genome = "";
        std::vector<std::string> ref_sequences = {};
        auto ref_duration = std::chrono::milliseconds(0);

        // Only load reference sequences if we have query sequences (not pre-computed embeddings)
        if (!is_precomputed_embeddings)
        {
            auto ref_start_time = std::chrono::high_resolution_clock::now();

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

            auto ref_end_time = std::chrono::high_resolution_clock::now();
            ref_duration = std::chrono::duration_cast<std::chrono::milliseconds>(ref_end_time - ref_start_time);
            std::cout << "[MAIN] Reference loading & formatting time: " << ref_duration.count() << " ms" << std::endl;
            std::cout << "[MAIN] Total Data loading time: " << (query_duration.count() + ref_duration.count()) << " ms" << std::endl
                    << std::endl;
        }
        else
        {
            std::cout << "[MAIN] Skipping reference loading (not needed for pre-computed embeddings)" << std::endl
                    << std::endl;
        }

        if (is_precomputed_embeddings)
        {
            std::cout << "[MAIN] Total Data loading time: " << (main_pipeline_time + ref_duration.count()) << " ms" << std::endl
                      << std::endl;
        }
        else
        {
            std::cout << "[MAIN] Total Data loading time: " << (query_duration.count() + ref_duration.count()) << " ms" << std::endl
                      << std::endl;
        }

        // Load search index
        const int dim = Config::Build::DIM;

        std::cout << "[MAIN] HNSW INDEX LOADING STEP" << std::endl;
        std::cout << "[MAIN] Search Index Config:" << std::endl;
        std::cout << "[MAIN] Index file: " << index_file << std::endl;
        std::cout << "[MAIN] Dimension: " << dim << std::endl;
        std::cout << "[MAIN] EF: " << ef << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
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

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "[MAIN] Finished loading index" << std::endl;
        std::cout << "[MAIN] Index loaded time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // Inference step - only if we have sequences to vectorize
        Vectorizer *vectorizer = nullptr; // Declare here

        if (!is_precomputed_embeddings)
        {
            std::cout << "[MAIN] INFERENCE STEP" << std::endl;
            vectorizer = new Vectorizer(model_path, batch_size, max_len, model_out_size);

            // Run vectorization
            start_time = std::chrono::high_resolution_clock::now();

            embeddings = vectorizer->vectorize(query_sequences);

            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            main_pipeline_time += duration.count();

            // Print results
            std::cout << "[MAIN] Inference completed" << std::endl;
            std::cout << "[MAIN] Inference time: " << duration.count() << " ms" << std::endl
                      << std::endl;
        }

        std::cout << "[MAIN] HNSW SEARCH STEP" << std::endl;
        std::cout << "[MAIN] Searching for nearest neighbors..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        //* Original HNSW search
        // auto [neighbors, distances] = search(alg_hnsw, embeddings);

        //* HNSWPQ search
        auto [neighbors, distances] = faiss_search(alg_hnsw, embeddings, k_clusters, ef);

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        main_pipeline_time += duration.count();
        std::cout << "[MAIN] HNSW Search completed" << std::endl;
        std::cout << "[MAIN] Search time: " << duration.count() << " ms" << std::endl
                  << std::endl;
        std::cout << "[MAIN] Neighbors found: " << neighbors.size() << std::endl;
        std::cout << "[MAIN] Number of neighbors/query: " << (neighbors.empty() ? 0 : neighbors[0].size()) << std::endl
                  << std::endl;

        int total_nei = 0;
        for (const auto &nei : neighbors)
            total_nei += nei.size();
        std::cout << "[MAIN] Total neighbors: " << total_nei << std::endl;
        std::cout << "[MAIN] Avg neighbors/query: " << (static_cast<float>(total_nei) / neighbors.size()) << std::endl
                  << std::endl;

        // Free-up search memory
        delete alg_hnsw;

        std::cout << "[MAIN] POST-PROCESSING STEP" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        // Declare variables outside the if-else blocks
        std::vector<std::string> final_seqs;
        std::vector<float> final_dists;
        std::vector<size_t> final_ids;
        std::vector<int> final_sw_scores; // For SW reranking

        // ONLY proceed with post-processing if we have sequence data
        //* L2 distance reranking
        if (!is_precomputed_embeddings)
        {
            if (use_dynamic)
            {
                if (use_streaming)
                {
                    std::cout << "[MAIN] Using STREAMING output to SAM file: " << sam_file << std::endl;
                    post_process_l2_dynamic_streaming(neighbors, distances, ref_genome, query_sequences, query_ids, ref_len, stride, k, embeddings, *vectorizer, k_clusters, sam_file, "ref");
                    // Skip the rest of the post-processing and output saving
                }
                else
                {
                    std::cout << "[MAIN] Using NORMAL output to SAM file: " << sam_file << std::endl;
                    std::tie(final_seqs, final_dists, final_ids) = post_process_l2_dynamic(neighbors, distances, ref_genome, query_sequences, ref_len, stride, k, embeddings, *vectorizer, k_clusters);
                }
            }
            else
            {
                std::tie(final_seqs, final_dists, final_ids) = post_process_l2_static(neighbors, distances, ref_sequences, query_sequences, ref_len, stride, k, embeddings, *vectorizer, k_clusters);
            }

            //* Smith-Waterman reranking
            // if (use_dynamic)
            // {
            //     std::tie(final_seqs, final_scores) = post_process_sw_dynamic(neighbors, distances, ref_genome, query_sequences, ref_len, stride, k, k_clusters);
            // }
            // else
            // {
            //     std::tie(final_seqs, final_sw_scores, final_ids) = post_process_sw_static(neighbors, distances, ref_sequences, query_sequences, ref_len, stride, k, rerank_lim);
            // }
        }
        else
        {
            std::cout << "[MAIN] Skipping post-processing (no sequence data available)" << std::endl;
        }

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        main_pipeline_time += duration.count();
        std::cout << "[MAIN] Post-processing completed" << std::endl;
        std::cout << "[MAIN] Post-processing time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // Print first 5 cands of first 3 queries for verification
        // for (size_t i = 0; i < std::min(size_t(3), query_sequences.size()); ++i)
        // {
        //     std::cout << "Query " << i << " - reranked candidates:" << std::endl;
        //     for (size_t j = 0; j < std::min(size_t(5), size_t(k)); ++j)
        //     {
        //         size_t idx = i * k + j;
        //         std::cout << "  Cand " << j << ": " << final_seqs[idx] << " (Distance: " << final_dists[idx] << ")" << std::endl;
        //     }
        // }

        // Save results to disk
        std::cout << "[MAIN] OUTPUT SAVING STEP" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        if (!use_streaming)
        {
            bool use_npy = true;
            std::string indices_file = output_dir + "/indices.npy";
            std::string distances_file = output_dir + "/distances.npy";

            if (stride == 1)
            {
                save_results(neighbors, distances, indices_file, distances_file, k, use_npy);
            }
            else
            {
                save_results(neighbors, distances, indices_file, distances_file, k_clusters, use_npy);
            }

            // Only write SAM if we have sequence data
            if (!is_precomputed_embeddings)
            {
                // Print length of final_seqs for verification
                // std::cout << "[MAIN] Total final sequences: " << final_seqs.size() << std::endl;
                // std::cout << "[MAIN] Total query sequences: " << query_sequences.size() << std::endl;
                // std::cout << "[MAIN] Total query IDs: " << query_ids.size() << std::endl;
                // std::cout << "[MAIN] Total candidate IDs: " << final_ids.size() << std::endl;

                // write_sam(final_seqs, final_dists, query_sequences, query_ids, final_ids, "ref", ref_len, k, sam_file);
            }
            else
            {
                std::cout << "[MAIN] Skipping SAM output (no sequence data available)" << std::endl;
            }

            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            main_pipeline_time += duration.count();
            std::cout << "[MAIN] Output saving time: " << duration.count() << " ms" << std::endl;
        }
        else
        {
            std::cout << "[MAIN] Skip normal output saving since streaming output is used." << std::endl;
        }

        auto master_end = std::chrono::high_resolution_clock::now();
        auto master_duration = std::chrono::duration_cast<std::chrono::milliseconds>(master_end - master_start);
        std::cout << "[MAIN] Finished processing file: " << query_input_file << std::endl;
        std::cout << "[MAIN] Index: " << index_prefix << std::endl;
        std::cout << "[MAIN] Total pipeline time: " << master_duration.count() << " ms" << std::endl;

        // Convert to seconds
        main_pipeline_time /= 1000;
        std::cout << "[MAIN] Main steps time: " << main_pipeline_time << " s" << std::endl;
        if (is_precomputed_embeddings)
        {
            std::cout << "[MAIN] Steps: Embeddings loading + Search + Output" << std::endl;
        }
        else
        {
            std::cout << "[MAIN] Steps: FASTQ loading + Inference + Search + Post-process + Output" << std::endl;
        }

        std::cout << "=== Pipeline Completed Successfully! ===" << std::endl;

        // Clean up
        if (vectorizer != nullptr)
        {
            delete vectorizer;
        }
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