#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <chrono>
#include <vector>
#include <omp.h>
#include "utils.hpp"
#include "vectorize.hpp"

/*
Create a representative training set by sampling evenly across the entire dataset.
Takes n_train vectors, usually 10% of total for optimal PQ codebook quality.
*/
std::vector<float> create_training_set(
    const std::vector<std::vector<float>> &all_embeddings,
    int n_train)
{

    int total_vectors = all_embeddings.size();
    int d = all_embeddings[0].size();

    // Calculate step size to cover entire dataset
    double step = static_cast<double>(total_vectors) / n_train;

    std::vector<float> train_data(n_train * d);

    for (int i = 0; i < n_train; ++i)
    {
        // Sample at evenly spaced intervals
        int sample_idx = static_cast<int>(i * step);

        // Ensure we don't go out of bounds
        sample_idx = std::min(sample_idx, total_vectors - 1);

        const auto &vec = all_embeddings[sample_idx];
        std::copy(vec.begin(), vec.end(),
                  train_data.begin() + i * d);
    }

    return train_data;
}

void build_faiss_index(const std::vector<std::vector<float>> &input_data,
                       const std::string &index_file,
                       int M_pq = 8, int nbits = 8, int M_hnsw = 16, int EFC = 200)
{
    // Build parameters
    int dim = input_data[0].size();
    size_t num_elements = input_data.size();

    // Validate input data
    if (num_elements == 0)
    {
        throw std::runtime_error("Input data is empty");
    }

    // Use 20% of data
    int n_train = static_cast<int>(num_elements * 0.2);

    std::cout << "[BUILD INDEX] Creating training set: " << n_train << " vectors from "
              << num_elements << " total vectors" << std::endl;

    // Create systematic training set
    std::vector<float> train_data = create_training_set(input_data, n_train);

    // Initialize IndexHNSWPQ
    faiss::IndexHNSWPQ index(dim, M_pq, M_hnsw, nbits);
    // Set build parameters
    index.hnsw.efConstruction = EFC;

    // Set multi-threading
    omp_set_num_threads(Config::Search::NUM_THREADS);

    std::cout << "[BUILD INDEX] Training PQ codebooks..." << std::endl;

    // Train the PQ quantizer
    auto train_start = std::chrono::high_resolution_clock::now();
    index.train(n_train, train_data.data());
    auto train_end = std::chrono::high_resolution_clock::now();
    auto train_duration = std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start);
    std::cout << "[BUILD INDEX] PQ training completed in " << train_duration.count() << " seconds" << std::endl;

    // Flatten input data for FAISS
    std::vector<float> vectors_flat(num_elements * dim);
    for (size_t i = 0; i < num_elements; ++i)
    {
        std::copy(input_data[i].begin(), input_data[i].end(), vectors_flat.begin() + i * dim);
    }

    std::cout << "[BUILD INDEX] Building HNSW graph with PQ compression..." << std::endl;

    // Hide cursor and create progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"building FAISS index"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    // Add vectors with progress tracking
    auto build_start = std::chrono::high_resolution_clock::now();

    // FAISS adds all vectors at once, so we simulate progress
    const size_t batch_size = 10000;
    for (size_t start = 0; start < num_elements; start += batch_size)
    {
        size_t end = std::min(start + batch_size, num_elements);
        size_t batch_count = end - start;

        // Add batch to index
        index.add(batch_count, vectors_flat.data() + start * dim);

        // Update progress bar
        size_t progress_percent = ((end) * 100) / num_elements;
        progressBar.set_progress(progress_percent);
    }

    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::seconds>(build_end - build_start);

    // Complete progress bar and show cursor
    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    std::cout << "[BUILD INDEX] HNSW graph built in " << build_duration.count() << " seconds" << std::endl;

    // Save index to file
    std::cout << "[BUILD INDEX] Saving index to: " << index_file << std::endl;
    faiss::write_index(&index, index_file.c_str());

    std::cout << "[BUILD INDEX] FAISS IndexHNSWPQ built and saved successfully!" << std::endl;
    std::cout << "[BUILD INDEX] Index parameters: M_pq=" << M_pq << ", nbits=" << nbits
              << ", M_hnsw=" << M_hnsw << ", efConstruction=" << EFC << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 7)
    {
        std::cerr << "Usage: " << argv[0] << " <ref_seq.txt> <search.index> [M_pq] [nbits] [M_hnsw] [EFC]" << std::endl;
        std::cerr << "  M_pq: number of PQ subquantizers (default: 8)" << std::endl;
        std::cerr << "  nbits: bits per subquantizer (default: 8)" << std::endl;
        std::cerr << "  M_hnsw: HNSW connectivity (default: 16)" << std::endl;
        std::cerr << "  EFC: efConstruction parameter (default: 200)" << std::endl;
        return 1;
    }

    std::string ref_file = argv[1];
    std::string index_file = argv[2];

    // Parse optional parameters with defaults
    // M_pq must be divisor of DIM, lower -> better accuracy
    // nbits must be 8, 10, or 12. Higher -> better accuracy
    int M_pq = (argc >= 4) ? std::stoi(argv[3]) : 8;
    int nbits = (argc >= 5) ? std::stoi(argv[4]) : 8;
    int M_hnsw = (argc >= 6) ? std::stoi(argv[5]) : 16;
    int EFC = (argc >= 7) ? std::stoi(argv[6]) : 200;

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
    std::cout << "[BUILD INDEX] Vectorization completed. Number of embeddings: " << embeddings.size() << std::endl;

    std::cout << "[BUILD INDEX] Building FAISS IndexHNSWPQ..." << std::endl;

    // Build FAISS index
    build_faiss_index(embeddings, index_file, M_pq, nbits, M_hnsw, EFC);

    return 0;
}