#include "vectorize.hpp"

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <sequences.txt> [output.npy]" << std::endl;
        return 1;
    }
    std::string sequences_file = argv[1];
    std::string output_file = (argc >= 3) ? argv[2] : "embeddings.npy";

    // read_file now returns (sequences, query_ids), we only need sequences
    auto [sequences, query_ids] = read_file(sequences_file);

    // Default config
    Vectorizer vectorizer;

    auto start = std::chrono::high_resolution_clock::now();
    auto embeddings = vectorizer.vectorize(sequences);
    auto end = std::chrono::high_resolution_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double microsec_per_query = (ms.count() * 1000.0) / sequences.size();

    std::cout << "E2E Inference Results:" << std::endl;
    std::cout << "Time: " << ms.count() << "ms, "
              << "Speed: " << microsec_per_query << " microsec/query" << std::endl;

    // Convert embeddings to format suitable for cnpy
    if (!embeddings.empty()) {
        size_t num_sequences = embeddings.size();
        size_t embedding_dim = embeddings[0].size();
        
        // Flatten the 2D vector into 1D for cnpy
        std::vector<float> flattened_embeddings;
        flattened_embeddings.reserve(num_sequences * embedding_dim);
        
        for (const auto& embedding : embeddings) {
            flattened_embeddings.insert(flattened_embeddings.end(), 
                                       embedding.begin(), embedding.end());
        }
        
        // Save as single numpy array (.npy) - more reliable for large files
        cnpy::npy_save(output_file, &flattened_embeddings[0], 
                      {num_sequences, embedding_dim});
        
        std::cout << "Embeddings saved to: " << output_file << std::endl;
        std::cout << "Shape: (" << num_sequences << ", " << embedding_dim << ")" << std::endl;
    }

    // Print first 10 values of first 5 embeddings
    for (size_t i = 0; i < std::min(embeddings.size(), size_t(5)); ++i)
    {
        std::cout << "Embed " << i << ": [";
        for (size_t j = 0; j < std::min(embeddings[i].size(), size_t(10)); ++j)
        {
            if (j > 0)
                std::cout << ", ";
            std::cout << embeddings[i][j];
        }
        std::cout << "...]" << std::endl;
    }

    return 0;
}