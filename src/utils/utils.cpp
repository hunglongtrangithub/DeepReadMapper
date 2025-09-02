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

/// @brief Wrapper function for reading input sequences efficiently.
/// @param file_path Path to the input file (FASTA, FASTQ, or plain text).
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX (for FASTA only).
/// @param stride Length of non-overlap part between 2 windows (for FASTA only, default: 1).
/// @return Vector of input sequences as strings
std::vector<std::string> read_file(const std::string &file_path, size_t ref_len, size_t stride)
{
    // Check file extension
    std::string file_ext = std::filesystem::path(file_path).extension().string();
    if (file_ext != ".fna" && file_ext != ".fasta" && file_ext != ".fa" && file_ext != ".fastq" && file_ext != ".txt")
    {
        throw std::runtime_error("Unsupported file format: " + file_ext + ". Only .fna/.fastq/.txt are supported.");
    }

    if (file_ext == ".fna" || file_ext == ".fasta" || file_ext == ".fa")
    {
        std::cout << "Detected FASTA file format." << std::endl;
        return preprocess_fasta(file_path, ref_len, stride);
    }
    else if (file_ext == ".fastq")
    {
        std::cout << "Detected FASTQ file format." << std::endl;
        return preprocess_fastq(file_path);
    }

    std::cout << "Detected plain text file format." << std::endl;

// For .txt or other plain text files, use mmap if available
#ifdef __linux__
    return read_txt_mmap(file_path); // Use memory mapping on Linux
#else
    return read_txt_default(file_path);
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
// Replace the save_results function with this simplified version:

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

int save_config(const std::unordered_map<std::string, ConfigValue> &config, const std::string &folder_path, const std::string &config_file)
{
    std::filesystem::create_directories(folder_path);

    std::string config_path = folder_path + "/" + config_file;

    std::ofstream out(config_file);
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