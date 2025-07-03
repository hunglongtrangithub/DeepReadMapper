#include "vectorize.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

std::vector<std::string> read_file(const std::string &file_path)
{
    /*
    Read sequences from a text file.
    */
    std::vector<std::string> sequences;
    std::ifstream file(file_path);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    std::string line;
    size_t line_count = 0;

    std::cout << "Reading sequences from: " << file_path << std::endl;

    while (std::getline(file, line))
    {
        // Remove any trailing whitespace/newlines
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
        {
            line.pop_back();
        }

        if (!line.empty())
        {
            sequences.push_back(line);
            line_count++;

            // Print progress every 1000 sequences
            if (line_count % 1000 == 0)
            {
                std::cout << "Read " << line_count << " sequences..." << std::endl;
            }
        }
    }

    file.close();
    std::cout << "Successfully read " << sequences.size() << " sequences" << std::endl;

    return sequences;
}

void analyze_input(const std::vector<std::string> &sequences)
{
    /*
    Calculate basic statistics on input sequences
    */
    if (sequences.empty())
        return;

    size_t min_len = sequences[0].length();
    size_t max_len = sequences[0].length();
    size_t total_len = 0;

    for (const auto &seq : sequences)
    {
        size_t len = seq.length();
        min_len = std::min(min_len, len);
        max_len = std::max(max_len, len);
        total_len += len;
    }

    double avg_len = static_cast<double>(total_len) / sequences.size();

    std::cout << "\n--- Sequence Statistics ---" << std::endl;
    std::cout << "Number of sequences: " << sequences.size() << std::endl;
    std::cout << "Min length: " << min_len << std::endl;
    std::cout << "Max length: " << max_len << std::endl;
    std::cout << "Average length: " << avg_len << std::endl;

    // Show first few sequences as examples
    std::cout << "\n--- Sample Sequences ---" << std::endl;
    for (size_t i = 0; i < std::min(static_cast<size_t>(3), sequences.size()); ++i)
    {
        std::string display_seq = sequences[i];
        if (display_seq.length() > 50)
        {
            display_seq = display_seq.substr(0, 50) + "...";
        }
        std::cout << "Seq " << i + 1 << " (len=" << sequences[i].length() << "): "
                  << display_seq << std::endl;
    }
}

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
        std::cout << "\n=== STEP 1: Reading Sequences from File ===" << std::endl;
        std::vector<std::string> sequences = read_file(sequences_file);

        if (sequences.empty())
        {
            std::cerr << "No sequences found in file!" << std::endl;
            return 1;
        }

        analyze_input(sequences);

        // Initialize vectorizer
        std::cout << "\n=== STEP 2: Initializing Vectorizer ===" << std::endl;
        Vectorizer vectorizer(model_path, batch_size, max_len, model_out_size);
        std::cout << "Vectorizer initialized successfully!" << std::endl;

        // Run vectorization
        std::cout << "\n=== STEP 3: Running Vectorization ===" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<float>> embeddings = vectorizer.vectorize(sequences);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Print results
        std::cout << "Vectorization completed in " << duration.count() << " ms" << std::endl;
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