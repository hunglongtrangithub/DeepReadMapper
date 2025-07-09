#include "config.hpp"
#include "utils.hpp"
#include "vectorize.hpp"
#include "hnswlib_dir/search.hpp"
#include <filesystem>

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <search.index> <query_seq.txt>" << std::endl;
        return 1;
    }

    try
    {
        std::cout << "=== DeepAligner CPU Pipeline ===" << std::endl;

        // Read from command line argument
        const std::string index_file = argv[1];
        const std::string sequences_file = argv[2];

        // Config inference parameters
        const std::string model_path = Config::Inference::MODEL_PATH;
        const size_t batch_size = Config::Inference::BATCH_SIZE;
        const size_t max_len = Config::Inference::MAX_LEN;
        const size_t model_out_size = Config::Inference::MODEL_OUT_SIZE;

        std::cout << "\nConfiguration:" << std::endl;
        std::cout << "Input file: " << sequences_file << std::endl;
        std::cout << "Model path: " << model_path << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Max sequence length: " << max_len << std::endl;
        std::cout << "Model output size: " << model_out_size << std::endl;

        // Read sequences from file
        std::cout << "\n[MAIN] Reading Sequences from File" << std::endl;
        std::vector<std::string> sequences = read_file(sequences_file);

        if (sequences.empty())
        {
            std::cerr << "No sequences found in file!" << std::endl;
            return 1;
        }

        analyze_input(sequences);

        // Load search index
        int dim = Config::Build::DIM;
        int ef = Config::Search::EF;
        std::cout << "[MAIN] Loading HNSW Index from: " << index_file << std::endl;
        if (!std::filesystem::exists(index_file))
        {
            throw std::runtime_error("Index file does not exist: " + index_file);
        }
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_file);

        alg_hnsw->setEf(ef);

        // Initialize vectorizer
        std::cout << "\n[MAIN] Start inference" << std::endl;
        Vectorizer vectorizer(model_path, batch_size, max_len, model_out_size);

        // Run vectorization
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<float>> embeddings = vectorizer.vectorize(sequences);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Print results
        std::cout << "[MAIN] Inference completed in " << duration.count() << " ms" << std::endl;
        std::cout << "Processing speed: " << (sequences.size() / duration.count())
                  << " query/ms" << std::endl;

        std::cout << "\n--- Embedding Results ---" << std::endl;
        std::cout << "Number of embeddings: " << embeddings.size() << std::endl;
        if (embeddings.empty())
        {
            throw std::runtime_error("No embeddings generated from input sequences!");
        }
        std::cout << "Embedding dimension: " << embeddings[0].size() << std::endl;

        // Print first 3 embeddings (or fewer if less available)
        size_t num_to_print = std::min(static_cast<size_t>(3), embeddings.size());
        size_t values_to_show = std::min(static_cast<size_t>(10), embeddings[0].size());

        for (size_t i = 0; i < num_to_print; ++i)
        {
            std::cout << "\nEmbedding " << i + 1 << " (first 10 values): ";
            for (size_t j = 0; j < values_to_show; ++j)
            {
                std::cout << embeddings[i][j] << " ";
            }
            if (embeddings[i].size() > 10)
            {
                std::cout << "...";
            }
            std::cout << std::endl;
        }

        std::cout << "[MAIN] Starting HNSW Search" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        auto [neighbors, distances] = search(alg_hnsw, embeddings);

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[MAIN] HNSW Search completed (not implemented)" << std::endl;
        std::cout << "Search duration: " << duration.count() << " ms" << std::endl;

        // Print top results
        std::cout << "\n--- Search Results (neighbor_id, distance) ---" << std::endl;
        size_t neighbors_to_show = std::min(static_cast<size_t>(10), neighbors[0].size());

        for (size_t i = 0; i < num_to_print; ++i)
        {
            std::cout << "Query " << i + 1 << ": ";
            for (size_t j = 0; j < neighbors_to_show; ++j)
            {

                std::cout << "(" << neighbors[i][j] << ", " << distances[i][j] << ")";
                if (j < neighbors[i].size() - 1)
                {
                    std::cout << ", ";
                }
            }
            if (neighbors[i].size() > neighbors_to_show)
            {
                std::cout << ", ...";
            }
            std::cout << std::endl;
        }

        std::cout << "\n=== Pipeline Completed Successfully! ===" << std::endl;

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