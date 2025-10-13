#include "cnpy.h"
#include "config.hpp"
#include "utils.hpp"
#include "post_processor.hpp"
#include "vectorize.hpp"
#include "parse_inputs.hpp"
#include <filesystem>

void save_embeddings_npy(const std::vector<std::vector<float>> &embeddings, const std::string &filepath)
{
    if (embeddings.empty())
    {
        throw std::runtime_error("Cannot save empty embeddings");
    }

    size_t num_vectors = embeddings.size();
    size_t dim = embeddings[0].size();

    // Flatten 2D vector to 1D
    std::vector<float> flat_data;
    flat_data.reserve(num_vectors * dim);

    for (const auto &vec : embeddings)
    {
        flat_data.insert(flat_data.end(), vec.begin(), vec.end());
    }

    // Save as .npy file
    cnpy::npy_save(filepath, flat_data.data(), {num_vectors, dim}, "w");
}

int main()
{
    try
    {
        auto master_start = std::chrono::high_resolution_clock::now();
        std::cout << "=== DeepAligner Post-Processing STREAMING Test ===" << std::endl
                  << std::endl;

        // Input files
        // const std::string indices_file = "/home/tam/tam-store3/temp_output/ecoli_150_sparse/indices.npy";
        // const std::string distances_file = "/home/tam/tam-store3/temp_output/ecoli_150_sparse/distances.npy";
        // const std::string query_seqs_file = "/mnt/7T-ssdD/species-datasets/ecoli/ecoli_150/ecoli_150.fastq";
        // const std::string ref_seqs_file = "/mnt/7T-ssdD/species-datasets/fasta_source/ecoli.fna";

        const std::string indices_file = "/home/tam/tam-store3/real_data/results/chr13/indices.npy";
        const std::string distances_file = "/home/tam/tam-store3/real_data/results/chr13/distances.npy";
        const std::string query_seqs_file = "/mnt/4T-Samsung1/PacBio_realdata/fastq/HG002_PacBio-HiFi-Revio_20231031_48x_GRCh38-GIABv3.fastq";
        const std::string ref_seqs_file = "/mnt/4T-Samsung1/PacBio_realdata/fasta/GRCh38_GIABv3/chromosomes/chr13.fa";

        // Output file
        const std::string output_dir = "/home/tam/tam-store3/real_data/results/chr13";
        const std::string sam_file = output_dir + "/streaming_results.sam";

        // Parameters
        const int k = 10;
        const int k_clusters = 5;
        const std::string ref_name = "human_chr13";

        // Load index config
        std::string index_dir = "/home/tam/tam-store3/real_data/sparse_index/chr13";
        std::string config_file = index_dir + "/config.txt";

        // Config inference parameters
        const std::string model_path = Config::Inference::MODEL_PATH;
        const size_t batch_size = Config::Inference::BATCH_SIZE;
        const size_t max_len = Config::Inference::MAX_LEN;
        const size_t model_out_size = Config::Inference::MODEL_OUT_SIZE;

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
            std::cerr << "[TEST] Error: Config file not found. Terminate." << std::endl;
            return 1;
        }

        std::cout << "[TEST] TEST CONFIG:" << std::endl;
        std::cout << "[TEST] Indices file: " << indices_file << std::endl;
        std::cout << "[TEST] Distances file: " << distances_file << std::endl;
        std::cout << "[TEST] Query file: " << query_seqs_file << std::endl;
        std::cout << "[TEST] Reference file: " << ref_seqs_file << std::endl;
        std::cout << "[TEST] Output SAM file: " << sam_file << std::endl;
        std::cout << "[TEST] K: " << k << std::endl;
        std::cout << "[TEST] K_clusters: " << k_clusters << std::endl;
        std::cout << "[TEST] Ref length: " << ref_len << std::endl;
        std::cout << "[TEST] Stride: " << stride << std::endl;
        std::cout << "[TEST] Reference name: " << ref_name << std::endl
                  << std::endl;

        // ============================================================
        // LOAD NPY FILES
        // ============================================================
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

        // ============================================================
        // LOAD QUERY SEQUENCES (WITH IDs from FASTQ)
        // ============================================================
        std::cout << "[TEST] LOADING QUERY SEQUENCES" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        auto [query_sequences, query_ids] = read_file(query_seqs_file);

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

        std::cout << "[TEST] Loaded " << query_sequences.size() << " query sequences" << std::endl;
        std::cout << "[TEST] Loaded " << query_ids.size() << " query IDs" << std::endl;

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Query loading time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // ============================================================
        // LOAD REFERENCE GENOME (for dynamic lookup)
        // ============================================================
        std::cout << "[TEST] LOADING REFERENCE GENOME" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        std::string ref_genome = extract_FASTA_sequence(ref_seqs_file);

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Reference genome size: " << ref_genome.size() << " bp" << std::endl;
        std::cout << "[TEST] Reference loading time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // ============================================================
        // INITIALIZE VECTORIZER & VECTORIZE QUERIES
        // ============================================================
        std::cout << "[TEST] INITIALIZING VECTORIZER" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        Vectorizer vectorizer(model_path, batch_size, max_len, model_out_size);

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Vectorizer initialization time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // std::cout << "[TEST] VECTORIZING QUERIES" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<float>> query_embeddings = vectorizer.vectorize(query_sequences);

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Query embeddings: " << query_embeddings.size() << " x " << query_embeddings[0].size() << std::endl;
        std::cout << "[TEST] Vectorization time: " << duration.count() << " ms" << std::endl
                  << std::endl;

        // After query vectorization, save embeddings
        std::cout << "[TEST] SAVING QUERY EMBEDDINGS" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        save_embeddings_npy(query_embeddings, output_dir + "/query_embeddings.npy");

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Query embeddings saved in: " << duration.count() << " ms" << std::endl << std::endl;

        // Load pre-computed query embeddings instead of vectorizing
        // std::cout << "[TEST] LOADING PRE-COMPUTED QUERY EMBEDDINGS" << std::endl;
        // start_time = std::chrono::high_resolution_clock::now();

        // cnpy::NpyArray query_emb_arr = cnpy::npy_load(output_dir + "/query_embeddings.npy");
        // size_t num_vecs = query_emb_arr.shape[0];
        // size_t vec_dim = query_emb_arr.shape[1];

        // std::vector<std::vector<float>> query_embeddings(num_vecs);
        // float *emb_data = query_emb_arr.data<float>();

        // for (size_t i = 0; i < num_vecs; ++i)
        // {
        //     query_embeddings[i].resize(vec_dim);
        //     for (size_t j = 0; j < vec_dim; ++j)
        //     {
        //         query_embeddings[i][j] = emb_data[i * vec_dim + j];
        //     }
        // }

        // end_time = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        // std::cout << "[TEST] Loaded query embeddings: " << num_vecs << " x " << vec_dim << std::endl;
        // std::cout << "[TEST] Loading time: " << duration.count() << " ms" << std::endl
        //           << std::endl;
        
        // Self-implement step-by-step post_process_l2_dynamic_streaming to debug
        std::cout << "[TEST] Start debugging POST_PROCESS_L2_DYNAMIC_STREAMING" << std::endl;

        // ============================================================
        // STEP 1: COLLECT ALL SPARSE IDs (FLATTEN NEIGHBORS)
        // ============================================================
        std::cout << "[TEST] STEP 1: COLLECTING ALL SPARSE IDs" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        std::unordered_set<size_t> unique_sparse_ids_set;
        for (const auto &neighbor_list : neighbors)
        {
            for (size_t id : neighbor_list)
            {
                unique_sparse_ids_set.insert(id);
            }
        }
        
        std::vector<size_t> unique_sparse_ids(unique_sparse_ids_set.begin(), unique_sparse_ids_set.end());
        std::sort(unique_sparse_ids.begin(), unique_sparse_ids.end());
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Total sparse IDs collected: " << unique_sparse_ids.size() << std::endl;
        std::cout << "[TEST] Step 1 time: " << duration.count() << " ms" << std::endl << std::endl;
        
        // ============================================================
        // STEP 2: EXPAND SPARSE IDs TO DENSE IDs (SPARSE TRANSLATOR)
        // ============================================================
        std::cout << "[TEST] STEP 2: EXPANDING SPARSE IDs TO DENSE IDs" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        auto [ref_sequences, dense_ids, mapping_indices] = find_sequences(
            ref_genome,
            unique_sparse_ids,
            ref_len,
            stride
        );
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Reference sequences retrieved: " << ref_sequences.size() << std::endl;
        std::cout << "[TEST] Dense IDs: " << dense_ids.size() << std::endl;
        std::cout << "[TEST] Mapping indices: " << mapping_indices.size() << std::endl;
        std::cout << "[TEST] Step 2 time: " << duration.count() << " ms" << std::endl << std::endl;
        
        // ============================================================
        // STEP 3: VECTORIZE CANDIDATE SEQUENCES
        // ============================================================
        std::cout << "[TEST] STEP 3: VECTORIZING CANDIDATE SEQUENCES" << std::endl;
        std::cout << "[TEST] WARNING: This will take a long time! (~63M embeddings)" << std::endl;
        std::cout << "[TEST] Estimated time: " << (ref_sequences.size() / 1000) << " seconds" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<std::vector<float>> ref_embeddings = vectorizer.vectorize(ref_sequences);
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Reference embeddings: " << ref_embeddings.size() << " x " << ref_embeddings[0].size() << std::endl;
        std::cout << "[TEST] Step 3 time: " << duration.count() << " ms (" << (duration.count() / 1000.0) << " seconds)" << std::endl << std::endl;
        
        // ============================================================
        // STEP 4: SAVE REFERENCE EMBEDDINGS (FOR FUTURE USE)
        // ============================================================
        std::cout << "[TEST] STEP 4: SAVING REFERENCE EMBEDDINGS" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        save_embeddings_npy(ref_embeddings, output_dir + "/ref_embeddings.npy");
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Reference embeddings saved in: " << duration.count() << " ms" << std::endl << std::endl;
        
        // ============================================================
        // STEP 5: SAVE DENSE IDs AND MAPPING (FOR RECONSTRUCTION)
        // ============================================================
        std::cout << "[TEST] STEP 5: SAVING METADATA" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        // Save dense_ids
        std::vector<size_t> dense_ids_flat(dense_ids.begin(), dense_ids.end());
        cnpy::npy_save(output_dir + "/dense_ids.npy", dense_ids_flat.data(), {dense_ids_flat.size()}, "w");
        
        // Save mapping_indices
        std::vector<size_t> mapping_flat(mapping_indices.begin(), mapping_indices.end());
        cnpy::npy_save(output_dir + "/mapping_indices.npy", mapping_flat.data(), {mapping_flat.size()}, "w");
        
        // Save unique_sparse_ids
        std::vector<size_t> sparse_ids_flat(unique_sparse_ids.begin(), unique_sparse_ids.end());
        cnpy::npy_save(output_dir + "/unique_sparse_ids.npy", sparse_ids_flat.data(), {sparse_ids_flat.size()}, "w");
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TEST] Metadata saved in: " << duration.count() << " ms" << std::endl << std::endl;
        
        // ============================================================
        // DONE - STOP HERE FOR NOW
        // ============================================================
        std::cout << "[TEST] PHASE 1 COMPLETE!" << std::endl;
        std::cout << "[TEST] Next steps (to be implemented later):" << std::endl;
        std::cout << "  - Map sparse IDs from neighbors to dense IDs" << std::endl;
        std::cout << "  - Perform L2 reranking per query" << std::endl;
        std::cout << "  - Write to SAM file in batches" << std::endl << std::endl;
        
        // Skip the rest for now
        auto master_end = std::chrono::high_resolution_clock::now();
        auto master_duration = std::chrono::duration_cast<std::chrono::milliseconds>(master_end - master_start);
        
        std::cout << "[TEST] SUMMARY:" << std::endl;
        std::cout << "  - Total test time: " << master_duration.count() << " ms" << std::endl;
        std::cout << "  - Unique sparse IDs: " << unique_sparse_ids.size() << std::endl;
        std::cout << "  - Reference embeddings: " << ref_embeddings.size() << std::endl;
        std::cout << "  - Files saved in: " << output_dir << std::endl;
        std::cout << std::endl;
        
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