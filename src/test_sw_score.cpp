#include "post_processor.hpp"
#include <fstream>
#include <vector>

int main(int argc, char *argv[])
{
    if (argc == 3)
    {
        // Single pair mode (original functionality)
        std::string seq1 = argv[1];
        std::string seq2 = argv[2];

        int score = calc_sw_score(seq1, seq2);
        std::cout << "Smith-Waterman alignment score: " << score << std::endl;
        return 0;
    }
    else if (argc == 1)
    {
        // Batch mode to match Python script
        std::ifstream infile("/home/tam/tam-workspace/Research/DeepAligner-HNSW-CPU/unittest/test_data_quer.txt");
        if (!infile)
        {
            std::cerr << "Error: Could not open test data file" << std::endl;
            return 1;
        }

        std::vector<std::string> seqs;
        std::string line;
        while (std::getline(infile, line))
        {
            if (!line.empty())
            {
                seqs.push_back(line);
            }
        }
        infile.close();

        if (seqs.size() < 101)
        {
            std::cerr << "Error: Need at least 101 sequences for 100 pairs" << std::endl;
            return 1;
        }

        // Create 100 pairs: (seq[0], seq[1]), (seq[1], seq[2]), ..., (seq[99], seq[100])
        const int PAIR_NUM = 100;
        std::vector<int> scores;
        
        for (int i = 0; i < PAIR_NUM; ++i)
        {
            int score = calc_sw_score(seqs[i], seqs[i + 1]);
            scores.push_back(score);
        }

        // Save scores to file for comparison with Python
        std::ofstream outfile("/home/tam/tam-workspace/Research/tam-tools/temp/sw_scores_cpp.txt");
        if (!outfile)
        {
            std::cerr << "Error: Could not create output file" << std::endl;
            return 1;
        }

        for (int score : scores)
        {
            outfile << score << "\n";
        }
        outfile.close();

        std::cout << "Processed " << PAIR_NUM << " pairs and saved scores to sw_scores_cpp.txt" << std::endl;
        return 0;
    }
    else
    {
        std::cerr << "Usage: " << argv[0] << " <seq1> <seq2>  (for single pair)" << std::endl;
        std::cerr << "   or: " << argv[0] << "               (for batch processing)" << std::endl;
        return 1;
    }
}