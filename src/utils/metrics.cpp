#include "metrics.hpp"

int calc_sw_score_avx2(const std::string &seq1, const std::string &seq2)
{
    // TODO: Implement AVX2 optimized Smith-Waterman here
    // https://www.danysoft.com/estaticos/free/24%20-%20Case%20study%20-%20Pairwise%20sequence%20alignment%20with%20the%20Smith-Waterman%20algorithm.pdf
    return 0; // Placeholder
}

int calc_sw_score(const std::string &seq1, const std::string &seq2)
{
    // #ifdef __AVX2__
    //     // AVX2 optimized version here
    //     return calc_sw_score_avx2(seq1, seq2);
    // #else
    // Fallback scalar version
    // Scoring parameters to match Python implementation exactly
    const int match = 1;
    const int mismatch = -1;
    const int gap_penalty = -1; // gap penalty of 1 in Python = -1 score

    size_t len1 = seq1.size();
    size_t len2 = seq2.size();

    // Create DP matrix
    std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1, 0));
    int max_score = 0;

    // Fill DP matrix using standard Smith-Waterman with linear gap penalty
    for (size_t i = 1; i <= len1; ++i)
    {
        for (size_t j = 1; j <= len2; ++j)
        {
            int score_match = dp[i - 1][j - 1] + (seq1[i - 1] == seq2[j - 1] ? match : mismatch);
            int score_delete = dp[i - 1][j] + gap_penalty;
            int score_insert = dp[i][j - 1] + gap_penalty;

            dp[i][j] = std::max({0, score_match, score_delete, score_insert});
            max_score = std::max(max_score, dp[i][j]);
        }
    }

    return max_score;
    // #endif
}

float calc_l2_dist(const std::vector<float> &vec1, const std::vector<float> &vec2)
{
    if (vec1.size() != vec2.size())
    {
        throw std::invalid_argument("Vector sizes mismatch: " + std::to_string(vec1.size()) + " vs " + std::to_string(vec2.size()));
    }

    float sum = 0.0f;
    for (size_t i = 0; i < vec1.size(); ++i)
    {
        float diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}