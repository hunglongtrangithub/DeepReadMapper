#include "parse_inputs.hpp"
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

// FASTQ preprocessing using traditional file I/O
std::vector<std::string> preprocess_fastq_default(const std::string &fastq_file)
{
    std::cout << "Processing FASTQ file: " << fastq_file << std::endl;

    std::ifstream infile(fastq_file);
    if (!infile.is_open())
    {
        throw std::runtime_error("Failed to open file: " + fastq_file);
    }

    // Get file size for progress tracking
    struct stat file_stat;
    if (stat(fastq_file.c_str(), &file_stat) != 0)
    {
        throw std::runtime_error("Could not stat file: " + fastq_file);
    }
    size_t file_size = file_stat.st_size;

    // Estimate number of sequences (rough estimate: file_size / 200 for FASTQ)
    std::vector<std::string> sequences;

    sequences.reserve(file_size / 200 + 1000);

    // Setup progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"processing FASTQ sequences"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    std::string line;
    int line_number = 0;
    size_t bytes_processed = 0;
    size_t last_progress_update = 0;

    while (std::getline(infile, line))
    {
        bytes_processed += line.size() + 1; // +1 for newline

        if (line_number % 4 == 1)
        { // Sequence line (2nd line in every 4-line block)
            sequences.emplace_back(PREFIX + line + POSTFIX);
        }
        ++line_number;

        // Update progress bar
        if (bytes_processed - last_progress_update > 1024 * 1024)
        {
            size_t progress_percent = (bytes_processed * 100) / file_size;
            progressBar.set_progress(progress_percent);
            last_progress_update = bytes_processed;
        }
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    infile.close();
    std::cout << "Successfully processed " << sequences.size() << " sequences" << std::endl;
    return sequences;
}

// FASTQ preprocessing using memory mapping (Linux only)
std::vector<std::string> preprocess_fastq_mmap(const std::string &fastq_file)
{
    std::cout << "Processing FASTQ file: " << fastq_file << " (using mmap)" << std::endl;

    // Open file
    int fd = open(fastq_file.c_str(), O_RDONLY);
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

    // Memory map the file
    const char *data = static_cast<const char *>(
        mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));

    if (data == MAP_FAILED)
    {
        close(fd);
        throw std::runtime_error("Failed to mmap file: " + fastq_file);
    }

    // Estimate number of sequences
    std::vector<std::string> sequences;
    sequences.reserve(sb.st_size / 200 + 1000);

    // Setup progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"processing FASTQ sequences (mmap)"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    const char *current = data;
    const char *end = data + sb.st_size;
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
            progressBar.set_progress((bytes_processed * 100) / sb.st_size);
            last_progress_update = bytes_processed;
        }
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    // Cleanup
    munmap(const_cast<char *>(data), sb.st_size);
    close(fd);

    std::cout << "Successfully processed " << sequences.size() << " sequences" << std::endl;
    return sequences;
}

// Wrapper function for FASTQ preprocessing
std::vector<std::string> preprocess_fastq(const std::string &fastq_file)
{
#ifdef __linux__
    return preprocess_fastq_mmap(fastq_file);
#else
    return preprocess_fastq_default(fastq_file);
#endif
}