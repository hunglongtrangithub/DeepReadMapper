#include "hnswm/index.hpp"

void build(std::vector<std::vector<float>> ref_vecs, std::string index_file)
{
    // Parameters for HNSW
    uint32_t DIM = 128;            // Vector dimension (matches VECTOR_DIM)
    uint32_t EFC = 128;            // ef construction parameter
    uint32_t M = 64;               // Number of connections per node
    uint64_t maxNumNodes = 100000; // Maximum number of nodes

    std::cout << "[INDEX BUILD] Building HNSW index with parameters:" << std::endl;
    std::cout << "  Dimension: " << DIM << std::endl;
    std::cout << "  EFC: " << EFC << std::endl;
    std::cout << "  M: " << M << std::endl;
    std::cout << "  Max nodes: " << maxNumNodes << std::endl;

    HNSW hnsw(DIM, EFC, M, maxNumNodes);

    // Build the index
    std::cout << "Building HNSW index..." << std::endl;
    hnsw.buildIndex(ref_vecs);

    std::cout << "HNSW index built successfully!" << std::endl;

    // Print summary
    hnsw.summarize();

    // Save index to file
    std::cout << "Saving index to " << index_file << "..." << std::endl;
    hnsw.save(index_file);
}

int main(int argc, char *argv[])
{
    // Load ref file & index file from command line arguments
    if (argc != 3)
    {
        std::cerr << "[MAIN] Usage: " << argv[0] << " <ref_file.txt> <search.index>" << std::endl;
        return 1;
    }

    std::string ref_file = argv[1];
    std::string index_file = argv[2];

    std::cout << "[MAIN] Read reference sequences from: " << ref_file << std::endl;

    std::vector<std::string> sequences = read_file(ref_file);

    analyze_input(sequences);

    std::cout << "[MAIN] Start inference" << std::endl;
    Vectorizer vectorizer; // Use default params

    std::vector<std::vector<float>> ref_vecs = vectorizer.vectorize(sequences);
    std::cout << "[MAIN] Inference completed" << std::endl;

    // Start building the HNSW index with timer
    std::cout << "[MAIN] Start building HNSW index" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    build(ref_vecs, index_file);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "[MAIN] HNSW index built and saved to " << index_file << std::endl;
    std::cout << "[MAIN] Index build time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}