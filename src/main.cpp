#include "utils.hpp"
#include "vectorize.hpp"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <sequence_file.txt>" << std::endl;
        std::cerr << "Example: " << argv[0] << " sequences.txt" << std::endl;
        return 1;
    }

    try
    {
        std::cout << "=== DeepAligner CPU Pipeline ===" << std::endl;

        // Read from command line argument
        const std::string sequences_file = argv[1];

        // Config inference parameters
        const std::string model_path = "models/finetuned_sgn33-new-a-Apr6.xml";
        const size_t batch_size = 100;
        const size_t max_len = 123;
        const size_t model_out_size = 128;

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
        if (!embeddings.empty())
        {
            std::cout << "Embedding dimension: " << embeddings[0].size() << std::endl;

            // Print first 3 embeddings (or fewer if less available)
            size_t num_to_print = std::min(static_cast<size_t>(3), embeddings.size());
            for (size_t i = 0; i < num_to_print; ++i)
            {
                std::cout << "\nEmbedding " << i + 1 << " (first 10 values): ";
                size_t values_to_show = std::min(static_cast<size_t>(10), embeddings[i].size());
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
        }

        std::cout << "[MAIN] Starting HNSW Search" << std::endl;
        // TODO: Implement HNSW search here
        std::cout << "[MAIN] HNSW Search completed (not implemented)" << std::endl;

        std::cout << "\n=== Pipeline Completed Successfully! ===" << std::endl;
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