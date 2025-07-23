#include "utils.hpp"
#include "progressbar.h"

std::vector<std::string> read_file_default(const std::string &file_path)
{
    std::cout << "Reading sequences from: " << file_path << std::endl;
    
    // Get file size for progress and pre-allocation
    struct stat file_stat;
    if (stat(file_path.c_str(), &file_stat) != 0) {
        throw std::runtime_error("Could not stat file: " + file_path);
    }
    size_t file_size = file_stat.st_size;
    
    // Open file in binary mode
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
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
        indicators::option::ShowRemainingTime{true}
    };
    
    // Parse buffer in chunks
    const char* start = buffer.get();
    const char* end = start + file_size;
    const char* current = start;
    size_t bytes_processed = 0;
    size_t last_progress_update = 0;
    
    while (current < end) {
        // Find end of line
        const char* line_end = current;
        while (line_end < end && *line_end != '\n' && *line_end != '\r') {
            line_end++;
        }
        
        // Create string if line is not empty
        if (line_end > current) {
            sequences.emplace_back(current, line_end - current);
        }
        
        // Skip line ending characters
        current = line_end;
        while (current < end && (*current == '\n' || *current == '\r')) {
            current++;
        }
        
        // Update progress bar
        bytes_processed = current - start;
        if (bytes_processed - last_progress_update > 1024 * 1024) {
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


std::vector<std::string> read_file_mmap(const std::string &file_path)
{
    std::cout << "Reading sequences from: " << file_path << " (using mmap)" << std::endl;
    
    // Open file
    int fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Could not open file: " + file_path);
    }
    
    // Get file size
    struct stat file_stat;
    if (fstat(fd, &file_stat) == -1) {
        close(fd);
        throw std::runtime_error("Could not stat file: " + file_path);
    }
    size_t file_size = file_stat.st_size;
    
    // Memory map the file
    const char* data = static_cast<const char*>(
        mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0)
    );
    if (data == MAP_FAILED) {
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
        indicators::option::ShowRemainingTime{true}
    };
    
    // Parse memory-mapped data
    const char* current = data;
    const char* end = data + file_size;
    size_t bytes_processed = 0;
    size_t last_progress_update = 0;
    
    while (current < end) {
        const char* line_start = current;
        
        // Find end of line
        while (current < end && *current != '\n' && *current != '\r') {
            current++;
        }
        
        // Add sequence if not empty
        if (current > line_start) {
            sequences.emplace_back(line_start, current - line_start);
        }
        
        // Skip line endings
        while (current < end && (*current == '\n' || *current == '\r')) {
            current++;
        }
        
        // Update progress bar
        bytes_processed = current - data;
        if (bytes_processed - last_progress_update > 1024 * 1024) {
            progressBar.set_progress((bytes_processed * 100) / file_size);
            last_progress_update = bytes_processed;
        }
    }
    
    progressBar.set_progress(100);
    indicators::show_console_cursor(true);
    
    // Cleanup
    munmap(const_cast<char*>(data), file_size);
    close(fd);
    
    std::cout << "Successfully read " << sequences.size() << " sequences" << std::endl;
    return sequences;
}

std::vector<std::string> read_file(const std::string &file_path)
{
    /* Wrapper function for efficient file reading */
    #ifdef __linux__
        return read_file_mmap(file_path); // Use memory mapping on Linux
    #else
        return read_file_default(file_path);
    #endif
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
        std::cout << "Sequence " << (i + 1) << ": " << sequences[i] << std::endl;
    }
    std::cout << "------------------------\n"
              << std::endl;
}