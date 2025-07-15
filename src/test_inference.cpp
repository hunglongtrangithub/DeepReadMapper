#include "vectorize.hpp"

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <sequences.txt>" << std::endl;
        return 1;
    }
    std::string sequences_file = argv[1];

    std::vector<std::string> sequences = read_file(sequences_file);

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