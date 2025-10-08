#include "utils.hpp"
#include "parse_inputs.hpp"
#include "progressbar.h"

/// @brief Read input sequences from a plain text file (one sequence per line).
/// @param file_path Path to the input file.
/// @return Vector of input sequences as strings.
std::vector<std::string> read_txt_default(const std::string &file_path)
{
    std::cout << "Reading sequences from: " << file_path << std::endl;

    // Get file size for progress and pre-allocation
    struct stat file_stat;
    if (stat(file_path.c_str(), &file_stat) != 0)
    {
        throw std::runtime_error("Could not stat file: " + file_path);
    }
    size_t file_size = file_stat.st_size;

    // Open file in binary mode
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    // Read entire file into buffer at once
    std::unique_ptr<char[]> buffer(new char[file_size + 1]);
    file.read(buffer.get(), file_size);
    buffer[file_size] = '\0';
    file.close();

    // Estimate number of sequences (rough estimate: file_size / 150)
    std::vector<std::string> sequences;
    sequences.reserve(file_size / 150 + 1000); // Pre-allocate with buffer

    // Setup progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"parsing sequences"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    // Parse buffer in chunks
    const char *start = buffer.get();
    const char *end = start + file_size;
    const char *current = start;
    size_t bytes_processed = 0;
    size_t last_progress_update = 0;

    while (current < end)
    {
        // Find end of line
        const char *line_end = current;
        while (line_end < end && *line_end != '\n' && *line_end != '\r')
        {
            line_end++;
        }

        // Create string if line is not empty
        if (line_end > current)
        {
            sequences.emplace_back(current, line_end - current);
        }

        // Skip line ending characters
        current = line_end;
        while (current < end && (*current == '\n' || *current == '\r'))
        {
            current++;
        }

        // Update progress bar
        bytes_processed = current - start;
        if (bytes_processed - last_progress_update > 1024 * 1024)
        {
            size_t progress_percent = (bytes_processed * 100) / file_size;
            progressBar.set_progress(progress_percent);
            last_progress_update = bytes_processed;
        }
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    std::cout << "Successfully read " << sequences.size() << " sequences" << std::endl;
    return sequences;
}

/// @brief Read input sequences using memory mapping (Linux only).
/// @param file_path Path to the input file.
/// @return Vector of input sequences as strings.
std::vector<std::string> read_txt_mmap(const std::string &file_path)
{
    std::cout << "Reading sequences from: " << file_path << " (using mmap)" << std::endl;

    // Open file
    int fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1)
    {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    // Get file size
    struct stat file_stat;
    if (fstat(fd, &file_stat) == -1)
    {
        close(fd);
        throw std::runtime_error("Could not stat file: " + file_path);
    }
    size_t file_size = file_stat.st_size;

    // Memory map the file
    const char *data = static_cast<const char *>(
        mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (data == MAP_FAILED)
    {
        close(fd);
        throw std::runtime_error("Could not mmap file: " + file_path);
    }

    // Pre-allocate sequences vector
    std::vector<std::string> sequences;
    sequences.reserve(file_size / 150 + 1000);

    // Setup progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"parsing sequences (mmap)"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    // Parse memory-mapped data
    const char *current = data;
    const char *end = data + file_size;
    size_t bytes_processed = 0;
    size_t last_progress_update = 0;

    while (current < end)
    {
        const char *line_start = current;

        // Find end of line
        while (current < end && *current != '\n' && *current != '\r')
        {
            current++;
        }

        // Add sequence if not empty
        if (current > line_start)
        {
            sequences.emplace_back(line_start, current - line_start);
        }

        // Skip line endings
        while (current < end && (*current == '\n' || *current == '\r'))
        {
            current++;
        }

        // Update progress bar
        bytes_processed = current - data;
        if (bytes_processed - last_progress_update > 1024 * 1024)
        {
            progressBar.set_progress((bytes_processed * 100) / file_size);
            last_progress_update = bytes_processed;
        }
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    // Cleanup
    munmap(const_cast<char *>(data), file_size);
    close(fd);

    std::cout << "Successfully read " << sequences.size() << " sequences" << std::endl;
    return sequences;
}

/// @brief Wrapper function for reading FASTA/FNA files
/// @param file_path Path to the FASTA file
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX
/// @param stride Length of non-overlap part between 2 windows
/// @param lookup_mode If true, do not add PREFIX/POSTFIX to sequences
/// @return Pair of (sequences, positional labels as size_t)
std::pair<std::vector<std::string>, std::vector<size_t>> read_fasta_file(const std::string &file_path, size_t ref_len, size_t stride, bool lookup_mode)
{
    std::string file_ext = std::filesystem::path(file_path).extension().string();
    if (file_ext != ".fna" && file_ext != ".fasta" && file_ext != ".fa")
    {
        throw std::runtime_error("Expected FASTA format (.fna/.fasta/.fa), got: " + file_ext);
    }
    
    std::cout << "Detected FASTA file format." << std::endl;
    return preprocess_fasta(file_path, ref_len, stride, lookup_mode);
}

/// @brief Wrapper function for reading FASTQ files
/// @param file_path Path to the FASTQ file
/// @return Pair of (sequences with PREFIX/POSTFIX, query IDs from headers)
std::pair<std::vector<std::string>, std::vector<std::string>> read_fastq_file(const std::string &file_path)
{
    std::string file_ext = std::filesystem::path(file_path).extension().string();
    if (file_ext != ".fastq" && file_ext != ".fq")
    {
        throw std::runtime_error("Expected FASTQ format (.fastq/.fq), got: " + file_ext);
    }
    
    std::cout << "Detected FASTQ file format." << std::endl;
    return preprocess_fastq(file_path);
}

/// @brief Generic wrapper function for reading input sequences (auto-detects format)
/// @param file_path Path to the input file (FASTA, FASTQ, or plain text)
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX (for FASTA only)
/// @param stride Length of non-overlap part between 2 windows (for FASTA only, default: 1)
/// @param lookup_mode If true, do not add PREFIX/POSTFIX to sequences (for FASTA only, default: false)
/// @return Pair of (sequences, query IDs). For FASTA/TXT returns empty IDs, for FASTQ returns actual IDs
std::pair<std::vector<std::string>, std::vector<std::string>> read_file(const std::string &file_path, size_t ref_len, size_t stride, bool lookup_mode)
{
    std::string file_ext = std::filesystem::path(file_path).extension().string();
    if (file_ext != ".fna" && file_ext != ".fasta" && file_ext != ".fa" && file_ext != ".fastq" && file_ext != ".fq" && file_ext != ".txt")
    {
        throw std::runtime_error("Unsupported file format: " + file_ext + ". Only .fna/.fastq/.txt are supported.");
    }

    if (file_ext == ".fna" || file_ext == ".fasta" || file_ext == ".fa")
    {
        std::cout << "Detected FASTA file format." << std::endl;
        auto [seqs, labels] = preprocess_fasta(file_path, ref_len, stride, lookup_mode);
        // For FASTA in search context, we don't need labels, return empty IDs
        return {seqs, std::vector<std::string>()};
    }
    else if (file_ext == ".fastq" || file_ext == ".fq")
    {
        std::cout << "Detected FASTQ file format." << std::endl;
        return preprocess_fastq(file_path);  // Returns (sequences, query_ids)
    }

    std::cout << "Detected plain text file format." << std::endl;
#ifdef __linux__
    return {read_txt_mmap(file_path), std::vector<std::string>()};
#else
    return {read_txt_default(file_path), std::vector<std::string>()};
#endif
}

/// @brief Analyze input sequences and print statistics.
/// @param sequences Vector of input sequences as strings.
void analyze_input(const std::vector<std::string> &sequences)
{
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
        std::cout << "Sequence " << (i + 1) << ": " << sequences[i] << std::endl;
    }
    std::cout << "------------------------\n"
              << std::endl;
}

/// @brief Save search results to files.
/// @param neighbors Vector of vectors containing neighbor indices.
/// @param distances Vector of vectors containing L2 distances from query to neighbors.
/// @param indices_file Output filename for neighbor indices.
/// @param distances_file Output filename for distances.
/// @param k Number of nearest neighbors to save.
/// @param use_npy Whether to save results in .npy format (default: true).
/// @return 0 if successful.
//! Note, this function is deprecated as we return SAM instead of bin/npy files

int save_results(const std::vector<std::vector<size_t>> &neighbors, const std::vector<std::vector<float>> &distances, const std::string &indices_file, const std::string &distances_file, size_t k, const bool use_npy)
{
    if (use_npy)
    {
        // Save results in NumPy format for Python compatibility
        size_t n_rows = neighbors.size();

        // Flatten the 2D vectors into 1D arrays
        std::vector<size_t> host_indices(n_rows * k);
        std::vector<float> host_distances(n_rows * k);

        for (size_t i = 0; i < n_rows; ++i)
        {
            for (size_t j = 0; j < k; ++j)
            {
                host_indices[i * k + j] = neighbors[i][j];
                host_distances[i * k + j] = distances[i][j];
            }
        }

        cnpy::npy_save(indices_file, host_indices.data(), {static_cast<unsigned long>(n_rows), static_cast<unsigned long>(k)});
        cnpy::npy_save(distances_file, host_distances.data(), {static_cast<unsigned long>(n_rows), static_cast<unsigned long>(k)});

        return 0;
    }

    // Save results in binary format
    size_t n_rows = neighbors.size();

    // Write indices file
    std::ofstream indices_out(indices_file, std::ios::binary);
    if (!indices_out)
    {
        throw std::runtime_error("Could not create indices file: " + indices_file);
    }

    for (size_t i = 0; i < n_rows; ++i)
    {
        indices_out.write(reinterpret_cast<const char *>(neighbors[i].data()), k * sizeof(size_t));
    }
    indices_out.close();

    // Write distances file
    std::ofstream distances_out(distances_file, std::ios::binary);
    if (!distances_out)
    {
        throw std::runtime_error("Could not create distances file: " + distances_file);
    }

    for (size_t i = 0; i < n_rows; ++i)
    {
        distances_out.write(reinterpret_cast<const char *>(distances[i].data()), k * sizeof(float));
    }
    distances_out.close();

    return 0;
}

//! This function is deprecated as we return SAM instead of bin/npy files
int save_results(const std::vector<std::vector<long int>> &neighbors, const std::vector<std::vector<float>> &distances, const std::string &indices_file, const std::string &distances_file, size_t k, const bool use_npy)
{
    // Convert long int to size_t
    std::vector<std::vector<size_t>> neighbors_size_t(neighbors.size());
    for (size_t i = 0; i < neighbors.size(); ++i)
    {
        neighbors_size_t[i].assign(neighbors[i].begin(), neighbors[i].end());
    }

    // Call the existing function
    return save_results(neighbors_size_t, distances, indices_file, distances_file, k, use_npy);
}

void write_sam(const std::vector<std::string>& final_seqs, 
               const std::vector<float>& final_scores,
               const std::vector<std::string>& query_seqs,
               const std::vector<std::string>& query_ids,      // Query IDs from FASTQ headers
               const std::vector<size_t>& sequence_ids,        // Dense sequences ID, pairwise with final_seqs/final_scores
               const std::string& ref_name,
               const int ref_len,
               size_t k, 
               const std::string& output_file) {
    
    std::ofstream sam_file(output_file);
    
    // Write SAM header
    // @HD: Header line with version and sort order
    // @SQ: Reference sequence dictionary with sequence name and length
    sam_file << "@HD\tVN:1.0\tSO:unsorted\n";
    sam_file << "@SQ\tSN:" << ref_name << "\tLN:" << ref_len << "\n";
    
    // Calculate PREFIX/POSTFIX lengths for stripping
    const size_t prefix_len = strlen(PREFIX);
    const size_t postfix_len = strlen(POSTFIX);
    
    size_t total_queries = query_seqs.size();
    
    for (size_t i = 0; i < total_queries; ++i) {
        // Strip PREFIX and POSTFIX from query sequence for SAM output
        std::string clean_query = query_seqs[i];
        if (clean_query.length() > prefix_len + postfix_len) {
            clean_query = clean_query.substr(prefix_len, clean_query.length() - prefix_len - postfix_len);
        }
        
        for (size_t j = 0; j < k && (i * k + j) < final_seqs.size(); ++j) {
            
            size_t idx = i * k + j;
            size_t seq_id = sequence_ids[idx];
            
            if (i == 0 && j < 5) {
                std::cout << "Query 0, Cand " << j << ": seq_id=" << seq_id 
                        << ", genomic_pos=" << (seq_id/2) << std::endl;
            }
            // Derive position and reverse complement from ID
            size_t genomic_pos = seq_id / 2 + 1;  // +1 for 1-based SAM position
            bool is_reverse = (seq_id % 2 == 1);
            
            // FLAG: primary/secondary + reverse complement
            int flag = (j == 0) ? 0 : 256;  // Primary vs secondary
            if (is_reverse) flag |= 16;     // Add reverse complement flag

            // Pseudo fields (MapQ, CIGAR)
            int mapq = 60;  // Pseudo MAPQ
            // Use pseudo cigar: Assume all matches (M)
            // TODO: Implement real CIGAR - from SW ranker
            std::string cigar = std::to_string(clean_query.length()) + "M";
            
            // Use actual query ID from FASTQ if available, otherwise generate one
            std::string qname = (i < query_ids.size() && !query_ids[i].empty()) 
                                ? query_ids[i] 
                                : "S1/" + std::to_string(i+1) + "/0";
            
            sam_file << qname << "\t"                // QNAME (real from FASTQ)
                    << flag << "\t"                  // FLAG (real)
                    << ref_name << "\t"              // RNAME (real)
                    << genomic_pos << "\t"           // POS (real, 1-based)
                    << mapq << "\t"                  // MAPQ (pseudo)
                    << cigar << "\t"                 // CIGAR (pseudo, based on cleaned query length)
                    << "*\t0\t0\t"                   // RNEXT, PNEXT, TLEN
                    << clean_query << "\t"           // SEQ (cleaned - no PREFIX/POSTFIX)
                    << "*\n";                        // QUAL
        }
    }
    
    sam_file.close();
}

int save_config(const std::unordered_map<std::string, ConfigValue> &config, const std::string &folder_path, const std::string &config_file)
{
    std::filesystem::create_directories(folder_path);

    std::string config_path = folder_path + "/" + config_file;

    std::ofstream out(config_path);
    if (!out)
    {
        throw std::runtime_error("Could not create config file: " + config_file);
    }

    for (const auto &pair : config)
    {
        out << pair.first << ": ";
        if (std::holds_alternative<size_t>(pair.second))
        {
            out << std::get<size_t>(pair.second);
        }
        else if (std::holds_alternative<float>(pair.second))
        {
            out << std::get<float>(pair.second);
        }
        else if (std::holds_alternative<std::string>(pair.second))
        {
            out << std::get<std::string>(pair.second);
        }
        out << "\n";
    }

    out.close();
    return 0;
}

std::unordered_map<std::string, ConfigValue> load_config(const std::string &config_file)
{
    std::ifstream in(config_file);
    if (!in)
    {
        throw std::runtime_error("Could not open config file: " + config_file);
    }

    std::unordered_map<std::string, ConfigValue> config;
    std::string line;
    while (std::getline(in, line))
    {
        size_t delim_pos = line.find(':');
        if (delim_pos == std::string::npos)
            continue;

        std::string key = line.substr(0, delim_pos);
        std::string value_str = line.substr(delim_pos + 1);
        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value_str.erase(0, value_str.find_first_not_of(" \t"));
        value_str.erase(value_str.find_last_not_of(" \t") + 1);

        // Try to parse as size_t, then float, else string
        try
        {
            size_t idx;
            size_t value_size_t = std::stoull(value_str, &idx);
            if (idx == value_str.size())
            {
                config[key] = value_size_t;
                continue;
            }
        }
        catch (...)
        {
        }

        try
        {
            size_t idx;
            float value_float = std::stof(value_str, &idx);
            if (idx == value_str.size())
            {
                config[key] = value_float;
                continue;
            }
        }
        catch (...)
        {
        }

        config[key] = value_str;
    }

    in.close();
    return config;
}

int save_id_map(const std::vector<size_t> &labels, const std::string &folder_path, const std::string &mapping_file)
{
    std::filesystem::create_directories(folder_path);

    std::string mapping_path = folder_path + "/" + mapping_file;

    std::ofstream mapping(mapping_path, std::ios::binary);
    if (!mapping)
    {
        throw std::runtime_error("Could not create mapping file: " + mapping_file);
    }

    mapping.write(reinterpret_cast<const char *>(labels.data()),
                  labels.size() * sizeof(size_t));
    mapping.close();
    return 0;
}

std::vector<size_t> load_id_map(const std::string &mapping_file)
{
    std::ifstream mapping(mapping_file, std::ios::binary);
    if (!mapping)
    {
        throw std::runtime_error("Could not open mapping file: " + mapping_file);
    }

    // Get file size
    mapping.seekg(0, std::ios::end);
    size_t file_size = mapping.tellg();
    mapping.seekg(0, std::ios::beg);

    if (file_size % sizeof(size_t) != 0)
    {
        throw std::runtime_error("Mapping file size is not a multiple of size_t");
    }

    size_t num_labels = file_size / sizeof(size_t);
    std::vector<size_t> labels(num_labels);

    mapping.read(reinterpret_cast<char *>(labels.data()), file_size);
    mapping.close();
    return labels;
}