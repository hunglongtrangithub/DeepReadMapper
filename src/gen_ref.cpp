#include "parse_inputs.hpp"
#include "progressbar.h"

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input.fna> <ref_length> [output_path]" << std::endl;
        std::cerr << "Example: " << argv[0] << " ecoli_400.fna 400" << std::endl;
        std::cerr << "Example: " << argv[0] << " ecoli_400.fna 400 /path/to/output/" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];

    int ref_len;
    try
    {
        ref_len = std::stoi(argv[2]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: Invalid reference length: " << argv[2] << std::endl;
        return 1;
    }

    if (ref_len <= 0)
    {
        std::cerr << "Error: Reference length must be positive" << std::endl;
        return 1;
    }

    // Generate output filename
    std::filesystem::path input_path(input_file);
    std::string filename = input_path.stem().string() + "_ref.txt";

    std::string output_file;
    if (argc == 4)
    {
        // Custom output path provided
        std::filesystem::path output_path(argv[3]);
        output_file = output_path / filename;
    }
    else
    {
        // Use current working directory
        output_file = filename;
    }

    std::cout << "Input file: " << input_file << std::endl;
    std::cout << "Reference length: " << ref_len << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << std::endl;

    try
    {
        // Process FASTA file
        std::vector<std::string> sequences = preprocess_fasta(input_file, ref_len);

        if (sequences.empty())
        {
            std::cerr << "Error: No sequences generated from input file" << std::endl;
            return 1;
        }

        // Create output directory if it doesn't exist
        std::filesystem::path output_path_obj(output_file);
        if (output_path_obj.has_parent_path())
        {
            std::filesystem::create_directories(output_path_obj.parent_path());
        }

        // Write to output file with progress bar
        std::cout << "Writing " << sequences.size() << " sequences to " << output_file << std::endl;

        indicators::show_console_cursor(false);
        indicators::ProgressBar progressBar{
            indicators::option::BarWidth{80},
            indicators::option::PrefixText{"writing sequences"},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true}};

        std::ofstream outfile(output_file);
        if (!outfile)
        {
            std::cerr << "Error: Cannot create output file: " << output_file << std::endl;
            return 1;
        }

        size_t total = sequences.size();
        size_t last_update = 0;
        for (size_t i = 0; i < total; ++i)
        {
            outfile << sequences[i] << std::endl;
            // Update progress bar every 10,000 lines or at the end
            if (i - last_update > 10000 || i + 1 == total)
            {
                progressBar.set_progress(((i + 1) * 100) / total);
                last_update = i;
            }
        }
        progressBar.set_progress(100);
        indicators::show_console_cursor(true);

        outfile.close();
        std::cout << "Successfully created " << output_file << std::endl;
        std::cout << "Total sequences: " << sequences.size() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}