#include "parse_inputs.hpp"
#include "config.hpp"
#include "progressbar.h"

static const std::array<char, 128> comp_table = []
{
    std::array<char, 128> table{};
    table['A'] = 'T';
    table['T'] = 'A';
    table['C'] = 'G';
    table['G'] = 'C';
    table['N'] = 'N';
    return table;
}();

size_t estimate_token_count(const std::string &fasta_path, int token_len, size_t stride)
{
    // Get file size from the OS
    std::uintmax_t file_size = std::filesystem::file_size(fasta_path);

    if (file_size < 100)
    {
        return 0; // Too small to contain sequence data
    }

    // Subtract estimated header and formatting overhead (1 line header + newlines)
    std::uintmax_t estimated_bases = file_size - 100;

    // Must have at least token_len bases to produce one sliding window
    if (estimated_bases < static_cast<std::uintmax_t>(token_len))
    {
        return 0;
    }

    // Sliding window of stride s: generates (L - k + 1) tokens
    size_t num_windows = static_cast<size_t>((estimated_bases - token_len) / stride) + 1;

    // Each window yields both forward and reverse complement
    return num_windows * 2;
}

// Compute reverse complement
std::string reverse_complement(const std::string &seq)
{

    std::string rc;
    rc.reserve(seq.size());
    for (auto it = seq.rbegin(); it != seq.rend(); ++it)
    {
        rc.push_back(comp_table[static_cast<unsigned char>(*it)]);
    }
    return rc;
}

// FASTA file reading using traditional file I/O
std::pair<const char *, size_t> read_fasta_default(const std::string &fasta_file, std::unique_ptr<char[]> &buffer)
{
    std::cout << "Reading FASTA file: " << fasta_file << std::endl;

    std::ifstream infile(fasta_file, std::ios::binary);
    if (!infile)
    {
        throw std::runtime_error("Failed to open FASTA file: " + fasta_file);
    }

    // Get file size
    struct stat file_stat;
    if (stat(fasta_file.c_str(), &file_stat) != 0)
    {
        throw std::runtime_error("Could not stat file: " + fasta_file);
    }
    size_t file_size = file_stat.st_size;

    // Setup progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"reading FASTA file"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    // Allocate buffer and read entire file
    buffer = std::make_unique<char[]>(file_size + 1);

    // Read in chunks to show progress
    const size_t chunk_size = 1024 * 1024; // 1MB chunks
    size_t bytes_read = 0;
    size_t last_progress_update = 0;

    progressBar.set_progress(0);

    while (bytes_read < file_size)
    {
        size_t to_read = std::min(chunk_size, file_size - bytes_read);
        infile.read(buffer.get() + bytes_read, to_read);
        bytes_read += to_read;

        // Update progress bar
        if (bytes_read - last_progress_update > chunk_size)
        {
            progressBar.set_progress((bytes_read * 100) / file_size);
            last_progress_update = bytes_read;
        }
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    buffer[file_size] = '\0';
    infile.close();

    std::cout << "File read complete: " << file_size << " bytes" << std::endl;
    return {buffer.get(), file_size};
}

// FASTA file reading using memory mapping (Linux only)
std::pair<const char *, size_t> read_fasta_mmap(const std::string &fasta_file, int &fd)
{
    std::cout << "Reading FASTA file: " << fasta_file << " (using mmap)" << std::endl;

    // Open file
    fd = open(fasta_file.c_str(), O_RDONLY);
    if (fd == -1)
    {
        throw std::runtime_error("Failed to open FASTA file: " + fasta_file);
    }

    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1)
    {
        close(fd);
        throw std::runtime_error("Failed to get file size: " + fasta_file);
    }

    // Setup progress bar for mapping
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"mapping FASTA file (mmap)"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    progressBar.set_progress(0);

    // Memory map the file
    const char *data = static_cast<const char *>(
        mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));

    if (data == MAP_FAILED)
    {
        close(fd);
        indicators::show_console_cursor(true);
        throw std::runtime_error("Failed to mmap file: " + fasta_file);
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    std::cout << "File mapped complete: " << sb.st_size << " bytes" << std::endl;
    return {data, static_cast<size_t>(sb.st_size)};
}

// Wrapper function for FASTA file reading
std::pair<const char *, size_t> read_fasta(const std::string &fasta_file, std::unique_ptr<char[]> &buffer, int &fd)
{
#ifdef __linux__
    return read_fasta_mmap(fasta_file, fd);
#else
    return read_fasta_default(fasta_file, buffer);
#endif
}

std::string extract_FASTA_sequence(const std::string &fasta_file)
{
    // Find end of first line (header)

    std::unique_ptr<char[]> buffer;
    int fd = -1;

    // Step 1: Read file
    auto [data, data_size] = read_fasta(fasta_file, buffer, fd);

    // Step 2: Extract sequence
    const char *seq_start = data;
    while (seq_start < data + data_size && *seq_start != '\n')
    {
        seq_start++;
    }
    if (seq_start < data + data_size)
        seq_start++; // Skip newline

    std::string genome_sequence;
    genome_sequence.reserve(data_size); // Overestimate is fine

    // Extract clean sequence
    for (const char *ptr = seq_start; ptr < data + data_size; ++ptr)
    {
        char c = *ptr;
        if (std::isspace(c))
            continue;

        c = std::toupper(static_cast<unsigned char>(c));
        if (c == 'A' || c == 'T' || c == 'C' || c == 'G' || c == 'N')
        {
            genome_sequence.push_back(c);
        }
    }

    // Step 3: Cleanup
#ifdef __linux__
    if (fd != -1)
    {
        munmap(const_cast<char *>(data), data_size);
        close(fd);
    }
#endif

    return genome_sequence;
}

// Single-threaded FASTA data processing
std::pair<std::vector<std::string>, std::vector<size_t>> format_fasta(const char *data, size_t data_size, const std::string &fasta_file, size_t ref_len, size_t stride, bool lookup_mode)
{
    std::cout << "[FASTA] Processing FASTA data..." << std::endl;

    // Step 1: Extract all sequences from the FASTA file
    std::vector<std::string> sequences;
    std::string current_seq;
    current_seq.reserve(1024 * 1024); // Reserve 1MB initially

    const char *ptr = data;
    const char *end = data + data_size;
    bool in_sequence = false;

    while (ptr < end)
    {
        if (*ptr == '>')
        {
            // Save previous sequence if exists
            if (!current_seq.empty())
            {
                sequences.push_back(std::move(current_seq));
                current_seq.clear();
                current_seq.reserve(1024 * 1024);
            }
            // Skip header line
            while (ptr < end && *ptr != '\n')
                ptr++;
            if (ptr < end)
                ptr++; // Skip newline
            in_sequence = true;
            continue;
        }

        if (in_sequence)
        {
            char c = *ptr;
            if (!std::isspace(c))
            {
                c = std::toupper(static_cast<unsigned char>(c));
                if (c == 'A' || c == 'T' || c == 'C' || c == 'G' || c == 'N')
                {
                    current_seq.push_back(c);
                }
            }
        }
        ptr++;
    }

    // Add last sequence
    if (!current_seq.empty())
    {
        sequences.push_back(std::move(current_seq));
    }

    std::cout << "[FASTA] Extracted " << sequences.size() << " sequences" << std::endl;

    // Step 2: Calculate total windows across all sequences
    size_t raw_windows = 0;
    for (const auto &seq : sequences)
    {
        if (seq.size() >= ref_len)
        {
            raw_windows += (seq.size() - ref_len) / stride + 1;
        }
    }
    size_t total_window = raw_windows * 2; // Forward + Reverse complement

    double mem_usage = (static_cast<double>(total_window) * (ref_len + (lookup_mode ? 0 : 2))) / (1024.0 * 1024.0);

    std::cout << "[FASTA] Raw windows across all sequences: " << raw_windows << std::endl;
    std::cout << "[FASTA] Total estimated sequences (forward + reverse complement): " << total_window << std::endl;
    std::cout << "[FASTA] Estimated output sequences: " << total_window << std::endl;
    std::cout << "[FASTA] Estimated RAM usage: " << std::fixed << std::setprecision(2) << mem_usage << " MB" << std::endl;

    // Step 3: Preallocate result vectors
    std::vector<std::string> result;
    std::vector<size_t> labels;
    result.reserve(total_window);
    labels.reserve(total_window);

    // Setup progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"processing FASTA windows"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    size_t sequences_processed = 0;
    size_t global_position = 0;

    // Step 4: Generate sliding windows for each sequence
    for (const auto &seq : sequences)
    {
        if (seq.size() < ref_len)
        {
            sequences_processed++;
            continue;
        }
    
        // Calculate number of windows for this sequence
        size_t num_windows = (seq.size() - ref_len) / stride + 1;
        
        // Direct sliding window on the sequence
        for (size_t i = 0; i < num_windows; ++i)
        {
            size_t start_pos = i * stride;
            
            // Extract window directly using substr (or string_view in C++17)
            std::string window = seq.substr(start_pos, ref_len);
            std::string rev = reverse_complement(window);
            std::string forward;
            std::string reverse;
    
            if (!lookup_mode)
            {
                forward.reserve(2 + ref_len);
                forward.append(PREFIX).append(window).append(POSTFIX);
    
                reverse.reserve(2 + ref_len);
                reverse.append(PREFIX).append(rev).append(POSTFIX);
            }
            else
            {
                forward = window;
                reverse = rev;
            }
    
            result.push_back(forward);
            result.push_back(reverse);
    
            labels.push_back((global_position << 1) | 0);
            labels.push_back((global_position << 1) | 1);
    
            global_position += stride;
        }
    
        sequences_processed++;
        progressBar.set_progress((sequences_processed * 100) / sequences.size());
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    std::cout << "[FASTA] Successfully processed " << result.size() << " windows from " << sequences.size() << " sequences" << std::endl;
    return {result, labels};
}

std::pair<std::vector<std::string>, std::vector<size_t>> format_fasta_batch(
    const char *data,
    size_t data_size,
    const std::string &fasta_file,
    size_t ref_len,
    size_t stride,
    bool lookup_mode,
    size_t batch_size,
    size_t &resume_pos,
    size_t &position_counter,
    std::string &buffer_state,
    size_t &buf_start_state,
    bool &is_complete)
{
    std::cout << "[FASTA-BATCH] Processing batch starting at byte " << resume_pos << "..." << std::endl;

    const char *seq_start;
    if (resume_pos == 0)
    {
        // First batch: skip header
        seq_start = data;
        while (seq_start < data + data_size && *seq_start != '\n')
        {
            seq_start++;
        }
        if (seq_start < data + data_size)
            seq_start++;
        resume_pos = seq_start - data;
    }
    else
    {
        seq_start = data + resume_pos;
    }

    std::vector<std::string> result;
    std::vector<size_t> labels;
    result.reserve(batch_size);
    labels.reserve(batch_size);

    std::string &buffer = buffer_state;
    if (buffer.capacity() == 0)
    {
        buffer.reserve(ref_len + std::max<int>(1024, stride));
    }
    size_t &buf_start = buf_start_state;
    size_t &position = position_counter;

    size_t bytes_processed = 0;

    // Process data until batch is full or data exhausted
    for (const char *ptr = seq_start; ptr < data + data_size && result.size() < batch_size; ++ptr)
    {
        char c = *ptr;
        bytes_processed++;

        if (std::isspace(c))
            continue;

        c = std::toupper(static_cast<unsigned char>(c));
        if (c != 'A' && c != 'T' && c != 'C' && c != 'G' && c != 'N')
            continue;

        buffer.push_back(c);

        // Process as many windows as we can
        while (buffer.size() - buf_start >= ref_len && result.size() < batch_size)
        {
            std::string window = buffer.substr(buf_start, ref_len);
            std::string rev = reverse_complement(window);
            std::string forward;
            std::string reverse;

            if (!lookup_mode)
            {
                forward.reserve(2 + ref_len);
                forward.append(PREFIX).append(window).append(POSTFIX);
                reverse.reserve(2 + ref_len);
                reverse.append(PREFIX).append(rev).append(POSTFIX);
            }
            else
            {
                forward = window;
                reverse = rev;
            }

            result.push_back(forward);
            result.push_back(reverse);
            labels.push_back((position << 1) | 0);
            labels.push_back((position << 1) | 1);

            buf_start += stride;
            position += stride;
        }

        // Compact buffer
        size_t min_compact = std::max<size_t>(ref_len, 4096);
        if (buf_start >= min_compact || buf_start >= buffer.size() / 2)
        {
            buffer.erase(0, buf_start);
            buf_start = 0;
        }

        // Update resume position
        resume_pos = ptr - data + 1;
    }

    // Check if we've processed all data
    is_complete = (resume_pos >= data_size);

    std::cout << "[FASTA-BATCH] Batch complete: " << result.size() << " sequences" << std::endl;
    return {result, labels};
}

// SIMD-accelerated reverse complement for chunks of 32 bytes
void reverse_complement_simd(const char *src, char *dst, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        dst[i] = comp_table[static_cast<unsigned char>(src[len - 1 - i])];
    }
}

// Multi-threaded FASTA data processing with multi-sequence support
std::pair<std::vector<std::string>, std::vector<size_t>> format_fasta_mp(const char *data, size_t data_size, const std::string &fasta_file, size_t ref_len, size_t stride, bool lookup_mode)
{
    std::cout << "[FASTA] Processing FASTA data (parallel)..." << std::endl;

    // Step 1: Extract all sequences from the FASTA file (single-threaded)
    std::vector<std::string> sequences;
    std::string current_seq;
    current_seq.reserve(1024 * 1024);

    const char *ptr = data;
    const char *end = data + data_size;
    bool in_sequence = false;

    while (ptr < end)
    {
        if (*ptr == '>')
        {
            if (!current_seq.empty())
            {
                sequences.push_back(std::move(current_seq));
                current_seq.clear();
                current_seq.reserve(1024 * 1024);
            }
            while (ptr < end && *ptr != '\n')
                ptr++;
            if (ptr < end)
                ptr++;
            in_sequence = true;
            continue;
        }

        if (in_sequence)
        {
            char c = *ptr;
            if (!std::isspace(c))
            {
                c = std::toupper(static_cast<unsigned char>(c));
                if (c == 'A' || c == 'T' || c == 'C' || c == 'G' || c == 'N')
                {
                    current_seq.push_back(c);
                }
            }
        }
        ptr++;
    }

    if (!current_seq.empty())
    {
        sequences.push_back(std::move(current_seq));
    }

    std::cout << "[FASTA] Extracted " << sequences.size() << " sequences" << std::endl;

    // Step 2: Calculate total windows across all sequences
    size_t raw_windows = 0;
    for (const auto &seq : sequences)
    {
        if (seq.size() >= ref_len)
        {
            raw_windows += (seq.size() - ref_len) / stride + 1;
        }
    }
    size_t total_windows = raw_windows * 2;

    double mem_usage = (static_cast<double>(total_windows) * (ref_len + (lookup_mode ? 0 : 2))) / (1024.0 * 1024.0);

    std::cout << "[FASTA] Raw windows across all sequences: " << raw_windows << std::endl;
    std::cout << "[FASTA] Total estimated sequences (forward + reverse): " << total_windows << std::endl;
    std::cout << "[FASTA] Estimated RAM usage: " << std::fixed << std::setprecision(2) << mem_usage << " MB" << std::endl;

    // Step 3: Create window descriptors for all sequences
    std::vector<WindowDescriptor> descriptors;
    descriptors.reserve(total_windows);

    size_t global_position = 0;
    size_t descriptor_idx = 0;

    for (const auto &seq : sequences)
    {
        if (seq.size() < ref_len)
            continue;

        size_t num_windows = (seq.size() - ref_len) / stride + 1;

        for (size_t i = 0; i < num_windows; ++i)
        {
            uint32_t pos = static_cast<uint32_t>(i * stride);
            descriptors.push_back({pos, 0, static_cast<uint32_t>(descriptor_idx++)}); // Forward
            descriptors.push_back({pos, 1, static_cast<uint32_t>(descriptor_idx++)}); // Reverse
            global_position += stride;
        }
    }

    // Step 4: Pre-allocate result vectors
    std::vector<std::string> result(total_windows);
    std::vector<size_t> labels(total_windows);

    // Step 5: Setup batching for parallel processing
    const size_t BATCH_SIZE = 1000;
    const size_t num_batches = (descriptors.size() + BATCH_SIZE - 1) / BATCH_SIZE;

    std::cout << "[FASTA] Processing " << descriptors.size() << " windows in " << num_batches << " batches" << std::endl;

    // Step 6: Setup progress tracking
    std::atomic<size_t> completed_batches{0};
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"processing FASTA windows (parallel)"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    const size_t prefix_len = lookup_mode ? 0 : strlen(PREFIX);
    const size_t postfix_len = lookup_mode ? 0 : strlen(POSTFIX);
    const char *prefix_ptr = lookup_mode ? nullptr : PREFIX;
    const char *postfix_ptr = lookup_mode ? nullptr : POSTFIX;

    // Step 7: Process batches in parallel
#pragma omp parallel for num_threads(Config::Preprocess::NUM_THREADS) schedule(dynamic, 1)
    for (size_t batch_id = 0; batch_id < num_batches; ++batch_id)
    {
        const size_t start_desc = batch_id * BATCH_SIZE;
        const size_t end_desc = std::min(start_desc + BATCH_SIZE, descriptors.size());
        const size_t batch_size = end_desc - start_desc;

        // Find which sequence each descriptor belongs to
        size_t seq_idx = 0;
        size_t cumulative_windows = 0;
        
        for (size_t i = start_desc; i < end_desc; ++i)
        {
            const WindowDescriptor &desc = descriptors[i];
            
            // Find the correct sequence for this descriptor
            while (seq_idx < sequences.size())
            {
                if (sequences[seq_idx].size() < ref_len)
                {
                    seq_idx++;
                    continue;
                }
                
                size_t seq_num_windows = (sequences[seq_idx].size() - ref_len) / stride + 1;
                size_t seq_total_windows = seq_num_windows * 2;
                
                if (i < cumulative_windows + seq_total_windows)
                    break;
                    
                cumulative_windows += seq_total_windows;
                seq_idx++;
            }
            
            const char *genome_data = sequences[seq_idx].data();
            const char *src = genome_data + desc.genome_pos;
            
            // Build window string
            const size_t out_len = prefix_len + ref_len + postfix_len;
            std::string window_str(out_len, '\0');
            char *dst = window_str.data();
            
            // Copy prefix
            if (prefix_len)
                std::memcpy(dst, prefix_ptr, prefix_len);
            
            if (desc.is_reverse == 0)
            {
                // Forward
                std::memcpy(dst + prefix_len, src, ref_len);
            }
            else
            {
                // Reverse complement
                reverse_complement_simd(src, dst + prefix_len, ref_len);
            }
            
            // Copy postfix
            if (postfix_len)
                std::memcpy(dst + prefix_len + ref_len, postfix_ptr, postfix_len);
            
            // Store results
            result[desc.result_idx] = std::move(window_str);
            labels[desc.result_idx] = (static_cast<size_t>(desc.genome_pos) << 1) | desc.is_reverse;
        }

        // Update progress
        size_t done = completed_batches.fetch_add(1) + 1;
        size_t percent = (done * 100) / num_batches;
        if (percent % 5 == 0)
        {
#pragma omp critical
            progressBar.set_progress(percent);
        }
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);
    
    std::cout << "[FASTA] Successfully processed " << result.size() << " windows from " << sequences.size() << " sequences" << std::endl;
    return {result, labels};
}

// Combined wrapper function that handles both I/O and processing
std::pair<std::vector<std::string>, std::vector<size_t>> preprocess_fasta(const std::string &fasta_file, size_t ref_len, size_t stride, bool lookup_mode)
{
    std::unique_ptr<char[]> buffer;
    int fd = -1;

    // Step 1: Read file
    auto [data, data_size] = read_fasta(fasta_file, buffer, fd);

    // Step 2: Process data
    auto [result, labels] = format_fasta(data, data_size, fasta_file, ref_len, stride, lookup_mode);

    //* Use multi-threaded version
    // TODO: Implement new multi-threaded version
    // auto [result, labels] = format_fasta_mp(data, data_size, fasta_file, ref_len, stride, lookup_mode);

    // Step 3: Cleanup
#ifdef __linux__
    if (fd != -1)
    {
        munmap(const_cast<char *>(data), data_size);
        close(fd);
    }
#endif

    return {result, labels};
}

// FASTQ file reading using traditional file I/O
std::pair<const char *, size_t> read_fastq_default(const std::string &fastq_file, std::unique_ptr<char[]> &buffer)
{
    std::cout << "Reading FASTQ file: " << fastq_file << std::endl;

    std::ifstream infile(fastq_file, std::ios::binary);
    if (!infile)
    {
        throw std::runtime_error("Failed to open FASTQ file: " + fastq_file);
    }

    // Get file size
    struct stat file_stat;
    if (stat(fastq_file.c_str(), &file_stat) != 0)
    {
        throw std::runtime_error("Could not stat file: " + fastq_file);
    }
    size_t file_size = file_stat.st_size;

    // Setup progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"reading FASTQ file"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    // Allocate buffer and read entire file
    buffer = std::make_unique<char[]>(file_size + 1);

    // Read in chunks to show progress
    const size_t chunk_size = 1024 * 1024; // 1MB chunks
    size_t bytes_read = 0;
    size_t last_progress_update = 0;

    progressBar.set_progress(0);

    while (bytes_read < file_size)
    {
        size_t to_read = std::min(chunk_size, file_size - bytes_read);
        infile.read(buffer.get() + bytes_read, to_read);
        bytes_read += to_read;

        // Update progress bar
        if (bytes_read - last_progress_update > chunk_size)
        {
            progressBar.set_progress((bytes_read * 100) / file_size);
            last_progress_update = bytes_read;
        }
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    buffer[file_size] = '\0';
    infile.close();

    std::cout << "File read complete: " << file_size << " bytes" << std::endl;
    return {buffer.get(), file_size};
}

// FASTQ file reading using memory mapping (Linux only)
std::pair<const char *, size_t> read_fastq_mmap(const std::string &fastq_file, int &fd)
{
    std::cout << "Reading FASTQ file: " << fastq_file << " (using mmap)" << std::endl;

    // Open file
    fd = open(fastq_file.c_str(), O_RDONLY);
    if (fd == -1)
    {
        throw std::runtime_error("Failed to open FASTQ file: " + fastq_file);
    }

    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1)
    {
        close(fd);
        throw std::runtime_error("Failed to get file size: " + fastq_file);
    }

    // Setup progress bar for mapping
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"mapping FASTQ file (mmap)"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    progressBar.set_progress(0);

    // Memory map the file
    const char *data = static_cast<const char *>(
        mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));

    if (data == MAP_FAILED)
    {
        close(fd);
        indicators::show_console_cursor(true);
        throw std::runtime_error("Failed to mmap file: " + fastq_file);
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    std::cout << "File mapped complete: " << sb.st_size << " bytes" << std::endl;
    return {data, static_cast<size_t>(sb.st_size)};
}

// Wrapper function for FASTQ file reading
std::pair<const char *, size_t> read_fastq(const std::string &fastq_file, std::unique_ptr<char[]> &buffer, int &fd)
{
#ifdef __linux__
    return read_fastq_mmap(fastq_file, fd);
#else
    return read_fastq_default(fastq_file, buffer);
#endif
}

// FASTQ data processing
std::pair<std::vector<std::string>, std::vector<std::string>> format_fastq(const char *data, size_t data_size, bool verbose)
{
    if (verbose)
    {
        std::cout << "Processing FASTQ data..." << std::endl;
    }

    // Count newlines to estimate sequences (Note that 'count' function is expensive)
    // size_t estimated_seqs = std::count(data, data + data_size, '\n') / 4;
    size_t estimated_seqs = data_size / 200; // Rough estimate: ~200 bytes per FASTQ record

    std::vector<std::string> sequences;
    std::vector<std::string> query_ids;
    sequences.reserve(estimated_seqs);
    query_ids.reserve(estimated_seqs);

    const char *current = data;
    const char *end = data + data_size;
    const size_t prefix_len = strlen(PREFIX);
    const size_t postfix_len = strlen(POSTFIX);

    // Setup progress bar only if verbose
    std::unique_ptr<indicators::ProgressBar> progressBar;
    if (verbose)
    {
        indicators::show_console_cursor(false);
        progressBar = std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{80},
            indicators::option::PrefixText{"processing FASTQ sequences"},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true});
        progressBar->set_progress(0);
    }

    int line_num = 0;
    size_t bytes_processed = 0;
    size_t last_progress_update = 0;
    const char *header_start = nullptr;

    while (current < end)
    {
        const char *line_start = current;

        // Find line end
        while (current < end && *current != '\n')
            current++;

        // Line 0: Header/ID (starts with @)
        if (line_num % 4 == 0)
        {
            header_start = line_start;
            // Skip the '@' character
            if (header_start < current && *header_start == '@')
            {
                header_start++;
            }
            // Extract query ID (everything until first space, tab, or slash)

            // Remove trailing newline if present (meaning forward/reverse indicator)
            const char *id_end = header_start;
            while (id_end < current && *id_end != ' ' && *id_end != '\t' && *id_end != '/')
            {
                id_end++;
            }
            query_ids.emplace_back(header_start, id_end - header_start);
        }
        // Line 1: Sequence
        else if (line_num % 4 == 1)
        {
            size_t seq_len = current - line_start;
            size_t total_len = prefix_len + seq_len + postfix_len;

            // Single allocation + memcpy (fastest)
            std::string result(total_len, '\0');
            char *dest = result.data();

            memcpy(dest, PREFIX, prefix_len);
            memcpy(dest + prefix_len, line_start, seq_len);
            memcpy(dest + prefix_len + seq_len, POSTFIX, postfix_len);

            sequences.emplace_back(std::move(result));
        }

        if (current < end)
            current++; // Skip \n
        line_num++;

        // Update progress bar only if verbose
        if (verbose && progressBar)
        {
            bytes_processed = current - data;
            if (bytes_processed - last_progress_update > 1024 * 1024)
            {
                progressBar->set_progress((bytes_processed * 100) / data_size);
                last_progress_update = bytes_processed;
            }
        }
    }

    if (verbose && progressBar)
    {
        progressBar->set_progress(100);
        indicators::show_console_cursor(true);
        std::cout << "Successfully processed " << sequences.size() << " sequences" << std::endl;
    }

    return {sequences, query_ids};
}

// FASTQ data processing using OpenMP for parallel processing
std::pair<std::vector<std::string>, std::vector<std::string>> format_fastq_mp(const char *data, size_t data_size)
{
    std::cout << "Processing FASTQ data..." << std::endl;

    const size_t num_threads = Config::Preprocess::NUM_THREADS;

    // Phase 1: Single-threaded chunking into FASTQ records
    std::vector<std::pair<size_t, size_t>> fastq_records; // (start, length) pairs

    const char *current = data;
    const char *end = data + data_size;

    while (current < end)
    {
        const char *record_start = current;

        // Skip 4 lines for each FASTQ record
        for (int i = 0; i < 4 && current < end; ++i)
        {
            while (current < end && *current != '\n')
                current++;
            if (current < end)
                current++; // Skip \n
        }

        size_t record_length = current - record_start;
        fastq_records.emplace_back(record_start - data, record_length);
    }

    std::cout << "Found " << fastq_records.size() << " complete FASTQ records" << std::endl;

    // Phase 2: Fixed chunk size processing
    std::vector<std::vector<std::string>> thread_seqs(num_threads);
    std::vector<std::vector<std::string>> thread_ids(num_threads);

    // Fixed chunk size (e.g., 1000 records per chunk)
    const size_t CHUNK_SIZE = Config::Preprocess::CHUNK_SIZE;
    const size_t num_chunks = (fastq_records.size() + CHUNK_SIZE - 1) / CHUNK_SIZE;

#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (size_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id)
    {
        size_t start_record = chunk_id * CHUNK_SIZE;
        size_t end_record = std::min(start_record + CHUNK_SIZE, fastq_records.size());

        size_t thread_id = omp_get_thread_num();

        // Process this fixed-size chunk
        for (size_t r = start_record; r < end_record; ++r)
        {
            const auto &[offset, length] = fastq_records[r];

            const char *record_data = data + offset;

            // Extract query ID from line 0 (header)
            const char *header_start = record_data;
            if (*header_start == '@')
                header_start++; // Skip '@'

            const char *header_end = header_start;
            while (header_end < record_data + length && *header_end != '\n' && *header_end != ' ' && *header_end != '\t' && *header_end != '/')
                header_end++;

            thread_ids[thread_id].emplace_back(header_start, header_end - header_start);

            // Skip to line 2 (sequence line)
            const char *line_start = record_data;
            while (line_start < record_data + length && *line_start != '\n')
                line_start++;
            if (line_start < record_data + length)
                line_start++; // Skip first \n

            // Find end of sequence line
            const char *line_end = line_start;
            while (line_end < record_data + length && *line_end != '\n')
                line_end++;

            // Build sequence with PREFIX/POSTFIX
            size_t seq_len = line_end - line_start;
            size_t prefix_len = strlen(PREFIX);
            size_t postfix_len = strlen(POSTFIX);
            size_t total_len = prefix_len + seq_len + postfix_len;

            std::string result(total_len, '\0');
            char *dest = result.data();

            memcpy(dest, PREFIX, prefix_len);
            memcpy(dest + prefix_len, line_start, seq_len);
            memcpy(dest + prefix_len + seq_len, POSTFIX, postfix_len);

            thread_seqs[thread_id].emplace_back(std::move(result));
        }
    }

    size_t total_seqs = 0;
    for (const auto &chunk : thread_seqs)
    {
        total_seqs += chunk.size();
    }

    std::vector<std::string> sequences;
    std::vector<std::string> query_ids;
    sequences.reserve(total_seqs);
    query_ids.reserve(total_seqs);

    for (auto &chunk : thread_seqs)
    {
        sequences.insert(sequences.end(),
                         std::make_move_iterator(chunk.begin()),
                         std::make_move_iterator(chunk.end()));
    }

    for (auto &chunk : thread_ids)
    {
        query_ids.insert(query_ids.end(),
                         std::make_move_iterator(chunk.begin()),
                         std::make_move_iterator(chunk.end()));
    }

    std::cout << "Successfully processed " << sequences.size() << " sequences (parallel)" << std::endl;
    return {sequences, query_ids};
}

// Combined wrapper function that handles both I/O and processing
std::pair<std::vector<std::string>, std::vector<std::string>> preprocess_fastq(const std::string &fastq_file)
{
    std::unique_ptr<char[]> buffer;
    int fd = -1;

    // Step 1: Read file
    auto start_time = std::chrono::high_resolution_clock::now();
    auto [data, data_size] = read_fastq(fastq_file, buffer, fd);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[FASTQ] File read time: " << duration.count() << " ms" << std::endl;

    // Step 2: Process data
    start_time = std::chrono::high_resolution_clock::now();

    auto [sequences, query_ids] = format_fastq(data, data_size, true);

    //* Format with multi-threads
    // TODO: Fix bug in multi-threaded version
    // auto [sequences, query_ids] = format_fastq_mp(data, data_size);

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[FASTQ] Formatting time: " << duration.count() << " ms" << std::endl;

    // Step 3: Cleanup
#ifdef __linux__
    if (fd != -1)
    {
        munmap(const_cast<char *>(data), data_size);
        close(fd);
    }
#endif

    return {sequences, query_ids};
}