#include "hnswlib_dir/index.hpp"

/*
Old code with single-thread HNSWLib
*/
void build_index(const std::vector<std::vector<float>> &input_data, const std::string &index_file)
{
    // Build parameters
    int dim = input_data[0].size();
    int M = Config::Build::GPH_DEG;
    int EFC = Config::Build::EFC;
    size_t num_elements = input_data.size();

    // Validate input data
    if (num_elements == 0)
    {
        throw std::runtime_error("Input data is empty");
    }

    // Init index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, num_elements, M, EFC);

    // Add data to index
    for (size_t i = 0; i < num_elements; i++)
    {
        alg_hnsw->addPoint(input_data[i].data(), i);
    }

    // Save index to file
    alg_hnsw->saveIndex(index_file);

    delete alg_hnsw;
    std::cout << "Index built and saved to: " << index_file << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <ref_seq.txt> <search.index>" << std::endl;
        return 1;
    }

    std::string ref_file = argv[1];
    std::string index_file = argv[2];

    // Config inference parameters
    const std::string model_path = Config::Inference::MODEL_PATH;
    const size_t batch_size = Config::Inference::BATCH_SIZE;
    const size_t max_len = Config::Inference::MAX_LEN;
    const size_t model_out_size = Config::Inference::MODEL_OUT_SIZE;

    // Load input data
    std::cout << "[BUILD INDEX] Reading sequences from file: " << ref_file << std::endl;
    std::vector<std::string> sequences = read_file(ref_file);

    if (sequences.empty())
    {
        std::cerr << "No sequences found in file: " << ref_file << std::endl;
        return 1;
    }

    std::cout << "[BUILD INDEX] Starting vectorizing sequences..." << std::endl;
    Vectorizer vectorizer(model_path, batch_size, max_len, model_out_size);
    std::vector<std::vector<float>> embeddings = vectorizer.vectorize(sequences);
    std::cout << "Vectorization completed. Number of embeddings: " << embeddings.size() << std::endl;

    std::cout << "[BUILD INDEX] Building HNSW index..." << std::endl;
    // Build index
    build_index(embeddings, index_file);
    std::cout << "[BUILD INDEX] HNSW index built successfully!" << std::endl;

    return 0;
}