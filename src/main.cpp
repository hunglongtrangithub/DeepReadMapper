#include "cnpy.h"
#include "config.hpp"
#include "utils.hpp"
#include "vectorize.hpp"
// #include "hnswlib_dir/search.hpp"
#include "hnswpq/search.hpp"
#include <filesystem>

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 8)
    {
        std::cerr << "Usage: " << argv[0] << " <search.index> <query_seq.txt> [EF] [K] [indices_output.npy] [distances_output.npy] [use_npy]" << std::endl;
        std::cerr << "  - EF: Optional HNSW search parameter (default: " << Config::Search::EF << ")" << std::endl;
        std::cerr << "  - K: Optional number of nearest neighbors to return (default: " << Config::Search::K << ")" << std::endl;

        std::cerr << "  - indices_output.npy: Optional indices output file (accept: indices.npy or indices.bin)" << std::endl;
        std::cerr << "  - distances_output.npy: Optional distances output file (accept: distances.npy or distances.bin)" << std::endl;
        std::cerr << "  - use_npy: Optional flag to save results in .npy format (default: false)" << std::endl;
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

        // Optional HNSW search parameters
        int ef = (argc >= 4) ? std::stoi(argv[3]) : Config::Search::EF;
        int k = (argc >= 5) ? std::stoi(argv[4]) : Config::Search::K;

        // Optional output file names with defaults
        const std::string indices_file = (argc >= 6) ? argv[5] : "indices.npy";
        const std::string distances_file = (argc >= 7) ? argv[6] : "distances.npy";

        const bool use_npy = (argc >= 6) ? std::string(argv[7]) == "true" : false;

        // const bool use_npy = (argc >= 6) ? std::string(argv[5]) == "true" : false;
        //TODO: Add option to use bin format
        const bool use_npy = true; // Always use npy for now

        // Config inference parameters
        const std::string model_path = Config::Inference::MODEL_PATH;
        const size_t batch_size = Config::Inference::BATCH_SIZE;
        const size_t max_len = Config::Inference::MAX_LEN;
        const size_t model_out_size = Config::Inference::MODEL_OUT_SIZE;

        std::cout << "[MAIN] PIPELINE CONFIG:" << std::endl;
        std::cout << "[MAIN] Input file: " << sequences_file << std::endl;
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
        std::vector<std::string> sequences = read_file(sequences_file);

        if (sequences.empty())
        {
            std::cerr << "No sequences found in input file!" << std::endl;
            return 1;
        }

        // analyze_input(sequences);

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
        faiss::Index* loaded_index = faiss::read_index(index_file.c_str());
        faiss::IndexHNSWPQ* alg_hnsw = dynamic_cast<faiss::IndexHNSWPQ*>(loaded_index);


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

        std::vector<std::vector<float>> embeddings = vectorizer.vectorize(sequences);

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

        // Save results using cnpy
        std::cout << "[MAIN] OUTPUT SAVING STEP" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        // Determine file extensions based on format
        std::string indices_output = indices_file;
        std::string distances_output = distances_file;

        save_results(neighbors, distances, indices_output, distances_output, Config::Search::K, use_npy);

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "[MAIN] Results saved to " << indices_output << " and " << distances_output << std::endl;
        std::cout << "[MAIN] Output saving time: " << duration.count() << " ms" << std::endl;

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