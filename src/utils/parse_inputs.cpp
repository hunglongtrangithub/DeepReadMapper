#include "parse_inputs.hpp"

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

size_t estimate_token_count(const std::string &fasta_path, int token_len)
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

    // Sliding window of stride 1: generates (L - k + 1) tokens
    size_t num_windows = estimated_bases - token_len + 1;

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

// FASTA preprocessing
std::vector<std::string> preprocess_fasta(const std::string &fasta_file)
{
    std::ifstream infile(fasta_file);
    if (!infile)
    {
        throw std::runtime_error("Failed to open FASTA file: " + fasta_file);
    }

    // Skip the first line (FASTA header)
    std::string header;
    std::getline(infile, header);

    // Preallocate vector
    size_t estimated_size = estimate_token_count(fasta_file, REFERENCE_SEQ_LEN);
    std::vector<std::string> result;
    result.reserve(estimated_size);

    std::string buffer;
    buffer.reserve(REFERENCE_SEQ_LEN);

    char c;
    while (infile.get(c))
    {
        if (std::isspace(c))
            continue;

        c = std::toupper(static_cast<unsigned char>(c));
        if (c != 'A' && c != 'T' && c != 'C' && c != 'G' && c != 'N')
            continue;

        buffer += c;

        if (buffer.size() >= REFERENCE_SEQ_LEN)
        {
            std::string window = buffer.substr(0, REFERENCE_SEQ_LEN);

            std::string forward;
            forward.reserve(2 + REFERENCE_SEQ_LEN);
            forward.append(PREFIX).append(window).append(POSTFIX);
            result.push_back(forward);

            std::string rev = reverse_complement(window);
            std::string reverse;
            reverse.reserve(2 + REFERENCE_SEQ_LEN);
            reverse.append(PREFIX).append(rev).append(POSTFIX);
            result.push_back(reverse);

            // Slide window by 1
            buffer.erase(0, 1);
        }
    }

    infile.close();
    return result;
}

std::vector<std::string> preprocess_fastq(const std::string &fastq_file)
{
    std::ifstream infile(fastq_file);
    if (!infile.is_open())
    {
        throw std::runtime_error("Failed to open file: " + fastq_file);
    }

    std::vector<std::string> sequences;
    std::string line;
    int line_number = 0;

    while (std::getline(infile, line))
    {
        if (line_number % 4 == 1)
        { // Sequence line (2nd line in every 4-line block)
            sequences.emplace_back(PREFIX + line + POSTFIX);
        }
        ++line_number;
    }

    infile.close();
    return sequences;
}