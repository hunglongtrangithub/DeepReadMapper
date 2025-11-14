#include "CLI11.hpp"
#include "progressbar.h"
#include "parse_inputs.hpp"
#include "utils.hpp"

int main(int argc, char *argv[])
{
    CLI::App app{"Universal Preprocessor for FASTA/FASTQ files"};

    // Required arguments
    std::string input_file;
    app.add_option("-i,--input", input_file, "Input FASTA or FASTQ file")
        ->required()
        ->check(CLI::ExistingFile);

    // Optional arguments with defaults
    int ref_len = 0;
    app.add_option("-l,--ref-len", ref_len, "Reference length (FASTA only)")
        ->check(CLI::NonNegativeNumber);

    size_t stride = 1;
    app.add_option("-s,--stride", stride, "Stride between windows (FASTA only)")
        ->check(CLI::PositiveNumber);

    std::string output_file;
    app.add_option("-o,--output", output_file, "Output file path (optional)");

    // Enable lookup mode during accuracy evaluation
    bool lookup_mode = false;
    app.add_flag("-L,--lookup", lookup_mode, "Lookup mode (no PREFIX/POSTFIX)");

    // Parse
    CLI11_PARSE(app, argc, argv);

    // Detect file type
    std::string file_ext = std::filesystem::path(input_file).extension().string();
    bool is_fastq = (file_ext == ".fastq" || file_ext == ".fq");
    bool is_fasta = (file_ext == ".fna" || file_ext == ".fasta" || file_ext == ".fa");
    
    if (!is_fastq && !is_fasta)
    {
        std::cerr << "Error: Unsupported file format: " << file_ext << std::endl;
        return 1;
    }

    // Validate ref_len for FASTA
    if (is_fasta && ref_len <= 0)
    {
        std::cerr << "Error: --ref-len must be positive for FASTA files" << std::endl;
        return 1;
    }

    // Generate output filename if not specified
    if (output_file.empty())
    {
        std::filesystem::path input_path(input_file);
        std::string suffix = "_" + std::to_string(ref_len);
        if (is_fastq) {
            suffix += "_quer.txt";
        } else {
            suffix += "_ref.txt";
        }

        output_file = input_path.stem().string() + suffix;
    }

    std::cout << "=== Universal Preprocessor ===" << std::endl;
    std::cout << "Input file: " << input_file << std::endl;
    std::cout << "File type: " << (is_fastq ? "FASTQ" : "FASTA") << std::endl;
    if (!is_fastq)
    {
        std::cout << "Reference length: " << ref_len << std::endl;
        std::cout << "Stride: " << stride << std::endl;
    }
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << std::endl;

    try
    {
        // Use unified read_file function
        auto [sequences, query_ids] = read_file(
            input_file, 
            ref_len,     // Only used for FASTA
            stride,      // Only used for FASTA
            lookup_mode        // lookup_mode = false -> (add PREFIX/POSTFIX)
        );

        if (sequences.empty())
        {
            std::cerr << "Error: No sequences generated from input file" << std::endl;
            return 1;
        }

        // Create output directory if needed
        std::filesystem::path output_path_obj(output_file);
        if (output_path_obj.has_parent_path())
        {
            std::filesystem::create_directories(output_path_obj.parent_path());
        }

        // Write sequences to output file with progress bar
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

        if (is_fasta)
            std::cout << "Total windows: " << sequences.size() << std::endl;

        
        // For FASTQ, also show query ID info
        if (is_fastq && !query_ids.empty())
        {
            std::cout << "Total sequences: " << sequences.size() << std::endl;
            std::cout << "Query IDs extracted: " << query_ids.size() << std::endl;
            std::cout << "Sample IDs (first 3):" << std::endl;
            for (size_t i = 0; i < std::min(size_t(3), query_ids.size()); ++i)
            {
                std::cout << "  " << query_ids[i] << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}