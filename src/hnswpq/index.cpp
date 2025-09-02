#include "hnswpq/index.hpp"

/// @brief Calculate estimated memory usage for IndexHNSWPQ
size_t estimate_memory(size_t num_vectors, size_t dim, int M_pq, int nbits, int M_hnsw, size_t n_train)
{
    size_t total_memory = 0;
    size_t training_memory = 0;

    // 1. PQ Codebooks memory
    // Each subquantizer has 2^nbits centroids, each centroid has (dim/M_pq) dimensions
    size_t centroids_per_subquantizer = 1ULL << nbits; // 2^nbits
    size_t dim_per_subquantizer = dim / M_pq;
    size_t pq_codebooks = M_pq * centroids_per_subquantizer * dim_per_subquantizer * sizeof(float);

    // 2. PQ codes for all vectors
    // Each vector encoded as M_pq bytes (one code per subquantizer)
    size_t pq_codes = num_vectors * M_pq * sizeof(uint8_t);

    // 3. HNSW graph structure
    // Each vector has ~M_hnsw connections per layer, with multiple layers
    // Estimate average connections per vector (accounting for layers)
    float avg_connections_per_vector = M_hnsw * 1.5f; // Approximation for multilayer
    size_t hnsw_graph = num_vectors * avg_connections_per_vector * sizeof(uint32_t);

    // 4. Vector IDs and metadata
    size_t metadata = num_vectors * sizeof(uint32_t);
    size_t training_peak = 0;

    // 5. Training memory (temporary during training phase)
    if (n_train > 0)
    {
        training_memory = n_train * dim * sizeof(float);                                             // Training vectors
        training_memory += M_pq * centroids_per_subquantizer * dim_per_subquantizer * sizeof(float); // Temp centroids during k-means
        training_memory += n_train * M_pq * sizeof(uint32_t);                                        // Assignment arrays during k-means

        training_peak = pq_codebooks + training_memory;

        std::cout << "[MEMORY ESTIMATE] Training (temporary): " << (training_memory / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "[MEMORY ESTIMATE] Peak Memory (during training): " << (training_peak / (1024 * 1024)) << " MB" << std::endl;
    }

    total_memory = pq_codebooks + pq_codes + hnsw_graph + metadata; // Final index size

    std::cout << "[MEMORY ESTIMATE] PQ Codebooks: " << (pq_codebooks / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "[MEMORY ESTIMATE] PQ Codes: " << (pq_codes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "[MEMORY ESTIMATE] HNSW Graph: " << (hnsw_graph / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "[MEMORY ESTIMATE] Metadata: " << (metadata / (1024 * 1024)) << " MB" << std::endl;

    std::cout << "[MEMORY ESTIMATE] Final Index Size: " << (total_memory / (1024 * 1024)) << " MB" << std::endl;

    return n_train > 0 ? std::max(training_peak, total_memory) : total_memory;
}

/// @brief Create a representative training set by sampling evenly across the entire dataset. Takes n_train vectors, usually 10% of total for optimal PQ codebook quality.

std::vector<float> create_training_set(
    const std::vector<std::vector<float>> &all_embeddings,
    size_t n_train)
{

    size_t total_vectors = all_embeddings.size();
    size_t d = all_embeddings[0].size();

    // Calculate step size to cover entire dataset
    double step = static_cast<double>(total_vectors) / n_train;

    std::vector<float> train_data(n_train * d);

    for (size_t i = 0; i < n_train; ++i)
    {
        // Sample at evenly spaced intervals
        size_t sample_idx = i * step;

        // Ensure we don't go out of bounds
        sample_idx = std::min(sample_idx, total_vectors - 1);

        const auto &vec = all_embeddings[sample_idx];
        std::copy(vec.begin(), vec.end(),
                  train_data.begin() + i * d);
    }

    return train_data;
}

void build_faiss_index(const std::vector<std::vector<float>> &input_data, const std::string &index_file, int M_pq, int nbits, int M_hnsw, int EFC)
{
    // Build parameters
    size_t dim = input_data[0].size();
    size_t num_elements = input_data.size();

    // Validate input data
    if (num_elements == 0)
    {
        throw std::runtime_error("Input data is empty");
    }

    // Take fraction of original data for training
    size_t n_train = num_elements * Config::Build::SAMPLE_RATE;

    std::cout << "[BUILD INDEX] Estimating memory usage..." << std::endl;
    estimate_memory(num_elements, dim, M_pq, nbits, M_hnsw, n_train);
    std::cout << std::endl;

    std::cout << "[BUILD INDEX] Creating training set: " << n_train << " / " << num_elements << " vectors (" << (Config::Build::SAMPLE_RATE * 100) << "%% dataset)" << std::endl;

    // Create training set
    std::vector<float> train_data = create_training_set(input_data, n_train);

    // Initialize IndexHNSWPQ
    faiss::IndexHNSWPQ index(dim, M_pq, M_hnsw, nbits);
    // Set build parameters
    index.hnsw.efConstruction = EFC;

    // Set multi-threading
    omp_set_num_threads(Config::Build::NUM_THREADS);

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
    if (argc < 5 || argc > 9)
    {
        std::cerr << "Usage: " << argv[0] << " <ref_seq.txt> <index_prefix> <ref_len> [stride] [M_pq] [nbits] [M_hnsw] [EFC]" << std::endl;
        std::cerr << "  stride: step size for sliding window (default: 1)" << std::endl;
        std::cerr << "  M_pq: number of PQ subquantizers (default: 8)" << std::endl;
        std::cerr << "  nbits: bits per subquantizer (default: 8)" << std::endl;
        std::cerr << "  M_hnsw: HNSW connectivity (default: 16)" << std::endl;
        std::cerr << "  EFC: efConstruction parameter (default: 200)" << std::endl;
        return 1;
    }

    const std::string ref_file = argv[1];
    const std::string index_prefix = argv[2];

    // Craft index file name and folder structure
    std::string basename = std::filesystem::path(index_prefix).filename().string();

    const std::string index_file = index_prefix + "/" + basename + ".index";

    const size_t ref_len = static_cast<size_t>(std::stoul(argv[3]));

    const size_t stride = (argc >= 5) ? static_cast<size_t>(std::stoul(argv[4])) : 1;

    // Parse optional parameters with defaults
    // M_pq must be divisor of DIM, lower -> better accuracy
    // nbits must be 8, 10, or 12. Higher -> better accuracy
    int M_pq = (argc >= 6) ? std::stoi(argv[5]) : 8;
    int nbits = (argc >= 7) ? std::stoi(argv[6]) : 8;
    int M_hnsw = (argc >= 8) ? std::stoi(argv[7]) : 16;
    int EFC = (argc >= 9) ? std::stoi(argv[8]) : 200;

    // Config inference parameters
    const std::string model_path = Config::Inference::MODEL_PATH;
    const size_t batch_size = Config::Inference::BATCH_SIZE;
    const size_t max_len = Config::Inference::MAX_LEN;
    const size_t model_out_size = Config::Inference::MODEL_OUT_SIZE;

    // Load input data
    std::cout << "[BUILD INDEX] Reading sequences from file: " << ref_file << std::endl;
    std::vector<std::string> sequences = read_file(ref_file, ref_len, stride);

    if (sequences.empty())
    {
        std::cerr << "No sequences found in file: " << ref_file << std::endl;
        return 1;
    }

    std::cout << "[BUILD INDEX] Starting vectorizing sequences..." << std::endl;
    Vectorizer vectorizer(model_path, batch_size, max_len, model_out_size);
    std::vector<std::vector<float>> embeddings = vectorizer.vectorize(sequences);
    std::cout << "[BUILD INDEX] Vectorization completed. Number of embeddings: " << embeddings.size() << std::endl;

    // Create and save config map
    std::cout << "[BUILD INDEX] Saving index config to folder: " << index_prefix << std::endl;
    std::unordered_map<std::string, ConfigValue> config = {
        {"index_type", std::string("HNSWPQ")},
        {"stride", stride},
        {"ref_len", ref_len},
        {"n_vects", static_cast<size_t>(embeddings.size())},
        {"dim", static_cast<size_t>(embeddings[0].size())},

        {"M_hnsw", static_cast<size_t>(M_hnsw)},
        {"EFC", static_cast<size_t>(EFC)},
        {"M_pq", static_cast<size_t>(M_pq)},
        {"nbits", static_cast<size_t>(nbits)},

        {"index_file", index_file},
    };

    save_config(config, index_prefix);
    std::cout << "[BUILD INDEX] Config saved successfully." << std::endl;

    std::cout << "[BUILD INDEX] Building FAISS IndexHNSWPQ..." << std::endl;

    // Build FAISS index
    build_faiss_index(embeddings, index_file, M_pq, nbits, M_hnsw, EFC);

    std::cout << "[BUILD INDEX] Finished index building process." << std::endl;

    return 0;
}