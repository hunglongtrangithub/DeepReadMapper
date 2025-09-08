#include "preprocess.hpp"
#include <string>
#include <algorithm>

Preprocessor::Preprocessor()
{
    uint8_t i = 0;
    for (auto &[token, index] : _Tok2Index)
    {
        if (hashToken(token[0], token[1], token[2]) != i)
        {
            throw std::runtime_error("Tokens are sorted incorrecly or are not fully defined");
        }
        tokens_.push_back(token);
        indices_.push_back(index);
        i++;
    }
}

std::vector<uint16_t> Preprocessor::preprocess(const std::string &seq, unsigned maxLen)
{
    unsigned len = std::min(maxLen, static_cast<unsigned int>(seq.size()));
    std::vector<uint16_t> result(len);

    char token0 = '<';
    char token1 = std::tolower(seq[0]);
    char token2 = std::tolower(seq[1]);
    result[0] = this->indices_[hashToken(token0, token1, token2)];
    unsigned i = 0;
    for (; i < len - 2; ++i)
    {
        token0 = std::tolower(seq[i]);
        token1 = std::tolower(seq[i + 1]);
        token2 = std::tolower(seq[i + 2]);
        result[i + 1] = this->indices_[hashToken(token0, token1, token2)];
    }
    token0 = std::tolower(seq[i++]);
    token1 = std::tolower(seq[i++]);
    token2 = (i < seq.size()) ? std::tolower(seq[i]) : '>';
    result[len - 1] = this->indices_[hashToken(token0, token1, token2)];
    return result;
}

std::vector<std::vector<uint16_t>> Preprocessor::preprocessBatch(const std::vector<std::string> &seqs, unsigned maxLen, bool verbose)
{
    std::vector<std::vector<uint16_t>> result(seqs.size());
    
    if (verbose) {
        // setup progress bar
        indicators::show_console_cursor(false);
        indicators::ProgressBar progressBar{
            indicators::option::BarWidth{80},
            indicators::option::PrefixText{"tokenizing + indexing"},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true}};
        
        for (size_t i = 0; i < seqs.size(); i++)
        {
            result[i] = this->preprocess(seqs[i], maxLen);
            // update progress bar
            if (i % 100000 == 0 || i == seqs.size() - 1)
            {
                float newProgressCompleted = static_cast<float>(i) / seqs.size() * 100;
                progressBar.set_progress(newProgressCompleted);
            }
        }
        // Close Current progress bar
        progressBar.mark_as_completed();
        indicators::show_console_cursor(true);
    } else {
        for (size_t i = 0; i < seqs.size(); i++)
        {
            result[i] = this->preprocess(seqs[i], maxLen);
        }
    }
    
    return result;
}