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

    // Find end of first line (header)
    const char *seq_start = data;
    while (seq_start < data + data_size && *seq_start != '\n')
    {
        seq_start++;
    }
    if (seq_start < data + data_size)
        seq_start++; // Skip newline

    // Preallocate vector
    size_t estimated_size = estimate_token_count(fasta_file, ref_len, stride);
    double mem_usage = (static_cast<double>(estimated_size) * 176.0) / (1024.0 * 1024.0);

    std::vector<std::string> result;
    std::vector<size_t> labels;

    result.reserve(estimated_size);
    labels.reserve(estimated_size);

    std::cout << "[FASTA] Estimated number of sequences: " << estimated_size << std::endl;
    std::cout << "[FASTA] Estimated RAM usage: " << std::fixed << std::setprecision(2) << mem_usage << " MB" << std::endl;

    std::string buffer;
    buffer.reserve(ref_len + std::max<int>(1024, stride));
    size_t buf_start = 0;
    size_t position = 0;

    // Setup progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"processing FASTA sequences"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    size_t bytes_processed = 0;
    size_t last_progress_update = 0;

    // Process the data
    for (const char *ptr = seq_start; ptr < data + data_size; ++ptr)
    {
        char c = *ptr;
        bytes_processed++;

        if (std::isspace(c))
            continue;

        c = std::toupper(static_cast<unsigned char>(c));
        if (c != 'A' && c != 'T' && c != 'C' && c != 'G' && c != 'N')
            continue;

        buffer.push_back(c);

        // Process as many windows as we can given current buffer contents
        while (buffer.size() - buf_start >= ref_len)
        {
            std::string window = buffer.substr(buf_start, ref_len);

            std::string rev = reverse_complement(window);
            std::string forward;
            std::string reverse;
            
            if (!lookup_mode) {
                forward.reserve(2 + ref_len);
                forward.append(PREFIX).append(window).append(POSTFIX);
                
                reverse.reserve(2 + ref_len);
                reverse.append(PREFIX).append(rev).append(POSTFIX);
            } else {
                forward = window;
                reverse = rev;
            }

            result.push_back(reverse);
            result.push_back(forward);

            labels.push_back((position << 1) | 0); // Forward
            labels.push_back((position << 1) | 1); // Reverse complement

            buf_start += stride;
            position += stride;
        }

        // Periodically compact the buffer to avoid unbounded growth / big memory
        // Update buffer with new content when ref_len is done, or when half buffered data is obsoleted
        size_t min_compact = std::max<size_t>(ref_len, 4096);
        if (buf_start >= min_compact || buf_start >= buffer.size() / 2)
        {
            buffer.erase(0, buf_start);
            buf_start = 0;
        }

        // Update progress bar
        if (bytes_processed - last_progress_update > 1024 * 1024)
        {
            progressBar.set_progress((bytes_processed * 100) / data_size);
            last_progress_update = bytes_processed;
        }
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    std::cout << "[FASTA] Successfully processed " << result.size() << " sequences" << std::endl;
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

// Thread-local window processor using descriptors
void process_window_batch_simd(
    const char *genome_data,
    const WindowDescriptor *descriptors,
    size_t batch_size,
    size_t ref_len,
    const char *prefix_ptr,
    const char *postfix_ptr,
    size_t prefix_len,
    size_t postfix_len,
    std::vector<std::string> &result,
    std::vector<size_t> &labels)
{
    const size_t out_len = prefix_len + ref_len + postfix_len;

    for (size_t i = 0; i < batch_size; ++i)
    {
        const WindowDescriptor &desc = descriptors[i];
        const char *src = genome_data + desc.genome_pos;

        // Pre-allocate result string
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
}

std::pair<std::vector<std::string>, std::vector<size_t>> format_fasta_mp(const char *data, size_t data_size, const std::string &fasta_file, size_t ref_len, size_t stride)
{
    //* Multi-threaded doesn't support lookup mode for now

    std::cout << "[FASTA] Start formatting data..." << std::endl;

    // Step 1: Extract genome sequence (single-threaded)
    const char *seq_start = data;
    while (seq_start < data + data_size && *seq_start != '\n')
        seq_start++;
    if (seq_start < data + data_size)
        seq_start++;

    std::string genome_sequence;
    genome_sequence.reserve(data_size);
    for (const char *ptr = seq_start; ptr < data + data_size; ++ptr)
    {
        char c = *ptr;
        if (std::isspace(c))
            continue;
        c = std::toupper(static_cast<unsigned char>(c));
        if (c == 'A' || c == 'T' || c == 'C' || c == 'G' || c == 'N')
            genome_sequence.push_back(c);
    }

    const size_t L = genome_sequence.size();
    std::cout << "[FASTA] Full sequence length: " << L << " bases" << std::endl;
    if (L < ref_len)
        return {{}, {}};

    const size_t num_windows = (L - ref_len) / stride + 1;
    const size_t total_sequences = num_windows * 2;

    // Step 2: Create descriptor array
    std::vector<WindowDescriptor> descriptors(total_sequences);

    // Populate descriptors (lightweight, no string allocation)
    for (size_t i = 0; i < num_windows; ++i)
    {
        uint32_t pos = static_cast<uint32_t>(i * stride);
        descriptors[i * 2] = {pos, 0, static_cast<uint32_t>(i * 2)};         // Forward
        descriptors[i * 2 + 1] = {pos, 1, static_cast<uint32_t>(i * 2 + 1)}; // Reverse
    }

    // Step 3: Pre-allocate result vectors
    std::vector<std::string> result;
    std::vector<size_t> labels;

    result.reserve(total_sequences);
    labels.reserve(total_sequences);

    // Step 4: Calculate processing batches for threads
    const size_t BATCH_SIZE = 1000;
    const size_t num_batches = (total_sequences + BATCH_SIZE - 1) / BATCH_SIZE;

    std::cout << "[FASTA] Total sequences: " << total_sequences << std::endl;
    std::cout << "[FASTA] Number of windows: " << num_windows << std::endl;
    std::cout << "[FASTA] Batch size: " << BATCH_SIZE << std::endl;

    // Step 5: Setup progress tracking
    std::atomic<size_t> completed_batches{0};
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"processing FASTA windows (parallel)"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    const char *g = genome_sequence.data();
    const size_t prefix_len = strlen(PREFIX);
    const size_t postfix_len = strlen(POSTFIX);
    const char *prefix_ptr = PREFIX;
    const char *postfix_ptr = POSTFIX;

    // Step 6: Process batches in parallel using descriptor API + SIMD
#pragma omp parallel for num_threads(Config::Build::NUM_THREADS) schedule(dynamic, 1)
    for (size_t batch_id = 0; batch_id < num_batches; ++batch_id)
    {
        const size_t start_desc = batch_id * BATCH_SIZE;
        const size_t end_desc = std::min(start_desc + BATCH_SIZE, total_sequences);
        const size_t batch_size = end_desc - start_desc;

        // Process this batch of descriptors with SIMD acceleration
        process_window_batch_simd(
            g,
            descriptors.data() + start_desc,
            batch_size,
            ref_len,
            prefix_ptr,
            postfix_ptr,
            prefix_len,
            postfix_len,
            result,
            labels);

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
    std::cout << "[FASTA] Successfully processed " << result.size() << " sequences" << std::endl;
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
    // auto [result, labels] = format_fasta_mp(data, data_size, fasta_file, ref_len, stride);

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
std::vector<std::string> format_fastq(const char *data, size_t data_size, bool verbose) {
    if (verbose) {
        std::cout << "Processing FASTQ data..." << std::endl;
    }
    
    // Count newlines to estimate sequences
    size_t estimated_seqs = std::count(data, data + data_size, '\n') / 4;
    
    std::vector<std::string> sequences;
    sequences.reserve(estimated_seqs);
    
    const char *current = data;
    const char *end = data + data_size;
    const size_t prefix_len = strlen(PREFIX);
    const size_t postfix_len = strlen(POSTFIX);
    
    // Setup progress bar only if verbose
    std::unique_ptr<indicators::ProgressBar> progressBar;
    if (verbose) {
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
    
    while (current < end) {
        const char *line_start = current;
        
        // Find line end
        while (current < end && *current != '\n') current++;
        
        if (line_num % 4 == 1) {
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
        
        if (current < end) current++; // Skip \n
        line_num++;
        
        // Update progress bar only if verbose
        if (verbose && progressBar) {
            bytes_processed = current - data;
            if (bytes_processed - last_progress_update > 1024 * 1024) {
                progressBar->set_progress((bytes_processed * 100) / data_size);
                last_progress_update = bytes_processed;
            }
        }
    }
    
    if (verbose && progressBar) {
        progressBar->set_progress(100);
        indicators::show_console_cursor(true);
        std::cout << "Successfully processed " << sequences.size() << " sequences" << std::endl;
    }
    
    return sequences;
}

// FASTQ data processing using OpenMP for parallel processing
std::vector<std::string> format_fastq_mp(const char *data, size_t data_size)
{
    std::cout << "Processing FASTQ data..." << std::endl;
    
    const size_t num_threads = Config::Search::NUM_THREADS;
    const size_t chunk_size = data_size / num_threads;
    
    std::vector<std::vector<std::string>> thread_results(num_threads);
    
    // Setup progress tracking
    std::atomic<size_t> completed_threads{0};
    size_t last_reported_percent = 0;
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"processing FASTQ sequences (parallel)"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};
    
    #pragma omp parallel for num_threads(num_threads)
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? data_size : (t + 1) * chunk_size;
        
        // Align to line boundaries
        if (start > 0) {
            while (start < data_size && data[start - 1] != '\n') start++;
        }
        if (end < data_size) {
            while (end < data_size && data[end] != '\n') end++;
        }
        
        // Reuse format_fastq for each chunk (without verbose to avoid progress conflicts)
        thread_results[t] = format_fastq(data + start, end - start, false);
        
        // Update progress every 10%
        size_t done = completed_threads.fetch_add(1) + 1;
        size_t current_percent = (done * 100) / num_threads;
        
        #pragma omp critical
        {
            // Only update if we've reached next 10% milestone
            if (current_percent >= last_reported_percent + 10 || current_percent == 100) {
                progressBar.set_progress(current_percent);
                last_reported_percent = (current_percent / 10) * 10; // Round down to nearest 10%
            }
        }
    }
    
    progressBar.set_progress(100);
    indicators::show_console_cursor(true);
    
    // Merge results
    size_t total_seqs = 0;
    for (const auto& chunk : thread_results) {
        total_seqs += chunk.size();
    }
    
    std::vector<std::string> sequences;
    sequences.reserve(total_seqs);
    
    for (auto& chunk : thread_results) {
        sequences.insert(sequences.end(), 
                        std::make_move_iterator(chunk.begin()),
                        std::make_move_iterator(chunk.end()));
    }
    
    std::cout << "Successfully processed " << sequences.size() << " sequences (parallel)" << std::endl;
    return sequences;
}

// Combined wrapper function that handles both I/O and processing
std::vector<std::string> preprocess_fastq(const std::string &fastq_file)
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
    // auto sequences = format_fastq(data, data_size, true);

    //* Format with multi-threads
    start_time = std::chrono::high_resolution_clock::now();
    auto sequences = format_fastq_mp(data, data_size);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[FASTQ] Formatting time (parallel): " << duration.count() << " ms" << std::endl;

    // Step 3: Cleanup
#ifdef __linux__
    if (fd != -1)
    {
        munmap(const_cast<char *>(data), data_size);
        close(fd);
    }
#endif

    return sequences;
}