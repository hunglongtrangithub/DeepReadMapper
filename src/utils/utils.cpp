#include "utils.hpp"
#include "progressbar.h"

std::vector<std::string> read_file(const std::string &file_path)
{
    /*
    Read sequences from a text file.
    */
    std::vector<std::string> sequences;
    std::ifstream file(file_path);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    std::string line;
    std::cout << "Reading sequences from: " << file_path << std::endl;

    // First pass: count total lines for progress bar
    size_t total_lines = 0;
    while (std::getline(file, line))
    {
        if (!line.empty())
        {
            total_lines++;
        }
    }

    // Reset file stream to beginning
    file.clear();
    file.seekg(0);

    // Hide cursor and create progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"reading sequences"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    size_t line_count = 0;
    while (std::getline(file, line))
    {
        // Remove any trailing whitespace/newlines
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
        {
            line.pop_back();
        }

        if (!line.empty())
        {
            sequences.push_back(line);
            line_count++;

            // Update progress bar
            size_t progress_percent = (line_count * 100) / total_lines;
            progressBar.set_progress(progress_percent);
        }
    }

    // Complete progress bar and show cursor
    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    file.close();
    std::cout << "Successfully read " << sequences.size() << " sequences" << std::endl;

    return sequences;
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