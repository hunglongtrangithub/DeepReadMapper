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

// Single-threaded FASTA data processing
std::pair<std::vector<std::string>, std::vector<size_t>> format_fasta(const char *data, size_t data_size, const std::string &fasta_file, size_t ref_len, size_t stride)
{
    std::cout << "Processing FASTA data..." << std::endl;

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

    std::cout << "Estimated number of sequences: " << estimated_size << std::endl;
    std::cout << "Estimated RAM usage: " << std::fixed << std::setprecision(2) << mem_usage << " MB" << std::endl;

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

            std::string forward;
            forward.reserve(2 + ref_len);
            forward.append(PREFIX).append(window).append(POSTFIX);
            result.push_back(forward);
            labels.push_back((position << 1) | 0); // Forward

            std::string rev = reverse_complement(window);
            std::string reverse;
            reverse.reserve(2 + ref_len);
            reverse.append(PREFIX).append(rev).append(POSTFIX);
            result.push_back(reverse);
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

    std::cout << "Successfully processed " << result.size() << " sequences" << std::endl;
    return {result, labels};
}

// Combined wrapper function that handles both I/O and processing
std::pair<std::vector<std::string>, std::vector<size_t>> preprocess_fasta(const std::string &fasta_file, size_t ref_len, size_t stride)
{
    std::unique_ptr<char[]> buffer;
    int fd = -1;

    // Step 1: Read file
    auto [data, data_size] = read_fasta(fasta_file, buffer, fd);

    // Step 2: Process data
    auto [result, labels] = format_fasta(data, data_size, fasta_file, ref_len, stride);

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
std::vector<std::string> format_fastq(const char *data, size_t data_size)
{
    std::cout << "Processing FASTQ data..." << std::endl;

    // Estimate number of sequences (rough estimate: file_size / 150 for FASTQ)
    std::vector<std::string> sequences;
    sequences.reserve(data_size / 150);

    // Setup progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"processing FASTQ sequences"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    const char *current = data;
    const char *end = data + data_size;
    int line_number = 0;
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

        // Process sequence line (2nd line in every 4-line block)
        if (line_number % 4 == 1 && current > line_start)
        {
            std::string line(line_start, current - line_start);
            sequences.emplace_back(PREFIX + line + POSTFIX);
        }

        // Skip line endings
        while (current < end && (*current == '\n' || *current == '\r'))
        {
            current++;
        }

        line_number++;

        // Update progress bar
        bytes_processed = current - data;
        if (bytes_processed - last_progress_update > 1024 * 1024)
        {
            progressBar.set_progress((bytes_processed * 100) / data_size);
            last_progress_update = bytes_processed;
        }
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    std::cout << "Successfully processed " << sequences.size() << " sequences" << std::endl;
    return sequences;
}

// FASTQ data processing using OpenMP for parallel processing
std::vector<std::string> format_fastq_mp(const char *data, size_t data_size)
{
    std::cout << "Processing FASTQ data (multi-threaded)..." << std::endl;

    // First pass: count lines and find sequence line positions
    std::vector<std::pair<const char *, size_t>> sequence_positions;
    const char *current = data;
    const char *end = data + data_size;
    int line_number = 0;

    // Setup progress bar for parsing
    indicators::show_console_cursor(false);
    indicators::ProgressBar parseBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"parsing FASTQ structure"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

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

        // Store sequence line positions (2nd line in every 4-line block)
        if (line_number % 4 == 1 && current > line_start)
        {
            sequence_positions.emplace_back(line_start, current - line_start);
        }

        // Skip line endings
        while (current < end && (*current == '\n' || *current == '\r'))
        {
            current++;
        }

        line_number++;

        // Update progress bar
        bytes_processed = current - data;
        if (bytes_processed - last_progress_update > 10 * 1024 * 1024) // Every 10MB
        {
            parseBar.set_progress((bytes_processed * 100) / data_size);
            last_progress_update = bytes_processed;
        }
    }

    parseBar.set_progress(100);
    indicators::show_console_cursor(true);

    std::cout << "Found " << sequence_positions.size() << " sequences" << std::endl;

    // Estimate and reserve space
    std::vector<std::string> sequences(sequence_positions.size());

    // Setup progress bar for processing
    indicators::show_console_cursor(false);
    indicators::ProgressBar processBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"processing FASTQ sequences (parallel)"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    size_t sequences_processed = 0;
    size_t last_seq_update = 0;

// Parallel processing of sequences
#pragma omp parallel for num_threads(Config::Search::NUM_THREADS) schedule(dynamic)

    for (size_t i = 0; i < sequence_positions.size(); ++i)
    {
        const auto &[line_start, line_length] = sequence_positions[i];
        std::string line(line_start, line_length);

        // Reserve space and construct the sequence
        std::string result;
        result.reserve(line_length + 2); // +2 for PREFIX and POSTFIX
        result.append(PREFIX).append(line).append(POSTFIX);

        sequences[i] = std::move(result);

// Update progress (thread-safe)
#pragma omp atomic
        sequences_processed++;

        // Update progress bar (only from one thread to avoid conflicts)
        if (omp_get_thread_num() == 0 &&
            sequences_processed - last_seq_update > 10000)
        {
            processBar.set_progress((sequences_processed * 100) / sequence_positions.size());
            last_seq_update = sequences_processed;
        }
    }

    processBar.set_progress(100);
    indicators::show_console_cursor(true);

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
    // auto sequences = format_fastq(data, data_size);

    //* Format with multi-threads
    start_time = std::chrono::high_resolution_clock::now();
    auto sequences = format_fastq_mp(data, data_size);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[FASTQ] Formatting time (multi-threaded): " << duration.count() << " ms" << std::endl;

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