#include "vectorize.hpp"
#include "utils.hpp"
#include "parse_inputs.hpp"

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 5)
    {
        std::cerr << "Usage: " << argv[0] << " <sequences.txt> <ref_len> [output.npy] [batch_size]" << std::endl;
        std::cerr << "  - batch_size: Number of sequences to process at once (default: 100000)" << std::endl;
        return 1;
    }
    std::string sequences_file = argv[1];
    int ref_len = std::stoi(argv[2]);
    std::string output_file = (argc >= 4) ? argv[3] : "embeddings.npy";
    size_t batch_size = (argc >= 5) ? std::stoull(argv[4]) : 100000;

    std::cout << "=== Batch Inference Configuration ===" << std::endl;
    std::cout << "Input file: " << sequences_file << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "Batch size: " << batch_size << " sequences" << std::endl << std::endl;

    // Initialize vectorizer
    Vectorizer vectorizer;
    size_t embedding_dim = 0;
    std::vector<std::vector<float>> all_embeddings;

    // Check file extension
    auto file_ext = std::filesystem::path(sequences_file).extension();
    
    if (file_ext == ".fasta" || file_ext == ".fna" || file_ext == ".fa")
    {
        std::cout << "Detected FASTA file format - using batch processing" << std::endl;
        
        // Read file into memory
        std::unique_ptr<char[]> buffer;
        int fd = -1;
        
        std::cout << "Reading FASTA file: " << sequences_file << " (using mmap)" << std::endl;
        fd = open(sequences_file.c_str(), O_RDONLY);
        if (fd == -1) {
            std::cerr << "Failed to open file" << std::endl;
            return 1;
        }
        
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            std::cerr << "Failed to get file size" << std::endl;
            return 1;
        }
        
        const char *data = static_cast<const char *>(
            mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
        
        if (data == MAP_FAILED) {
            close(fd);
            std::cerr << "Failed to mmap file" << std::endl;
            return 1;
        }
        
        size_t data_size = sb.st_size;
        std::cout << "File mapped: " << data_size << " bytes" << std::endl;
        
        // Estimate genome length and calculate accurate batch count
        std::cout << "\n[INFO] Analyzing genome..." << std::endl;
        const char *seq_start = data;
        while (seq_start < data + data_size && *seq_start != '\n')
            seq_start++;
        if (seq_start < data + data_size)
            seq_start++;
        
        size_t genome_length = 0;
        for (const char *ptr = seq_start; ptr < data + data_size; ++ptr) {
            char c = *ptr;
            if (std::isspace(c))
                continue;
            c = std::toupper(static_cast<unsigned char>(c));
            if (c == 'A' || c == 'T' || c == 'C' || c == 'G' || c == 'N')
                genome_length++;
        }
        
        // Calculate accurate sequence count
        size_t estimated_total_sequences = 0;
        size_t estimated_batches = 0;
        if (genome_length >= ref_len) {
            size_t num_windows = (genome_length - ref_len) + 1;  // stride = 1
            estimated_total_sequences = num_windows * 2;  // forward + reverse
            estimated_batches = (estimated_total_sequences + batch_size - 1) / batch_size;
        }
        
        std::cout << "[INFO] Genome length: " << genome_length << " bases" << std::endl;
        std::cout << "[INFO] Estimated sequences: " << estimated_total_sequences << std::endl;
        std::cout << "[INFO] Estimated batches: " << estimated_batches << std::endl;
        std::cout << "[INFO] Memory per batch: ~" << (batch_size * 176.0 / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "[INFO] Output file size: ~" << (estimated_total_sequences * 128 * sizeof(float) / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl << std::endl;
        
        // Batch processing state
        size_t resume_pos = 0;
        size_t position_counter = 0;
        std::string buffer_state;
        size_t buf_start_state = 0;
        bool is_complete = false;
        size_t total_sequences = 0;
        size_t batch_num = 0;
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        while (!is_complete)
        {
            batch_num++;
            std::cout << "\n=== Processing Batch " << batch_num << " / " << estimated_batches << " ===" << std::endl;
            
            // Get next batch of sequences from FASTA
            auto batch_start = std::chrono::high_resolution_clock::now();
            auto [batch_sequences, batch_labels] = format_fasta_batch(
                data, data_size, sequences_file, ref_len, 1, false,
                batch_size, resume_pos, position_counter,
                buffer_state, buf_start_state, is_complete);
            
            if (batch_sequences.empty()) {
                std::cout << "No more sequences to process" << std::endl;
                break;
            }
            
            auto batch_parse_end = std::chrono::high_resolution_clock::now();
            auto parse_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_parse_end - batch_start);
            std::cout << "[BATCH] Parsing time: " << parse_ms.count() << " ms" << std::endl;
            
            // Vectorize batch
            auto infer_start = std::chrono::high_resolution_clock::now();
            auto batch_embeddings = vectorizer.vectorize(batch_sequences);
            auto infer_end = std::chrono::high_resolution_clock::now();
            auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start);
            
            double microsec_per_query = (infer_ms.count() * 1000.0) / batch_sequences.size();
            std::cout << "[BATCH] Inference time: " << infer_ms.count() << " ms, "
                      << "Speed: " << microsec_per_query << " microsec/query" << std::endl;
            
            // Set embedding dimension from first batch
            if (embedding_dim == 0 && !batch_embeddings.empty()) {
                embedding_dim = batch_embeddings[0].size();
                std::cout << "[INFO] Detected embedding dimension: " << embedding_dim << std::endl;
            }
            
            // Accumulate embeddings
            all_embeddings.insert(all_embeddings.end(),
                                 std::make_move_iterator(batch_embeddings.begin()),
                                 std::make_move_iterator(batch_embeddings.end()));
            
            total_sequences += batch_sequences.size();
            size_t estimated_memory_mb = (all_embeddings.size() * embedding_dim * sizeof(float)) / (1024 * 1024);
            std::cout << "[INFO] Total processed: " << total_sequences << " sequences, "
                      << "Memory: ~" << estimated_memory_mb << " MB" << std::endl;
            
            // Clear batch data
            batch_sequences.clear();
            batch_sequences.shrink_to_fit();
            batch_embeddings.clear();
            batch_embeddings.shrink_to_fit();
        }
        
        // Cleanup
        munmap(const_cast<char *>(data), data_size);
        close(fd);
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
        std::cout << "\n=== Total Processing Time: " << total_ms.count() << " ms ===" << std::endl;
    }
    else if (file_ext == ".fastq" || file_ext == ".fq")
    {
        std::cout << "Detected FASTQ file format" << std::endl;
        auto [sequences, query_ids] = read_file(sequences_file);
        
        auto start = std::chrono::high_resolution_clock::now();
        all_embeddings = vectorizer.vectorize(sequences);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (!all_embeddings.empty()) {
            embedding_dim = all_embeddings[0].size();
        }
    }
    else
    {
        std::cerr << "Unsupported file format" << std::endl;
        return 1;
    }

    // Save embeddings
    std::cout << "\n=== Saving Embeddings ===" << std::endl;
    if (!all_embeddings.empty())
    {
        size_t num_sequences = all_embeddings.size();
        
        std::vector<float> flattened_embeddings;
        flattened_embeddings.reserve(num_sequences * embedding_dim);

        for (const auto &embedding : all_embeddings)
        {
            flattened_embeddings.insert(flattened_embeddings.end(),
                                        embedding.begin(), embedding.end());
        }

        cnpy::npy_save(output_file, &flattened_embeddings[0],
                       {num_sequences, embedding_dim});

        size_t file_size_mb = (num_sequences * embedding_dim * sizeof(float)) / (1024 * 1024);
        std::cout << "Embeddings saved to: " << output_file << std::endl;
        std::cout << "Shape: (" << num_sequences << ", " << embedding_dim << ")" << std::endl;
        std::cout << "File size: ~" << file_size_mb << " MB" << std::endl;
    }

    return 0;
}