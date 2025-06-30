#pragma once

#include <vector>
#include <stdexcept>
#include <cctype>
#include <immintrin.h>
#include "tok2index.hpp"
#include "progressbar.h"

static inline uint8_t char2Val(const char c)
{
    switch (c)
    {
    case 'a':
        return 0;
    case 'c':
        return 1;
    case 'g':
        return 2;
    case 't':
        return 3;
    default:
        return 7;
    }
}

// translate a token string into base 10 hash
// A -> 0, C ->1,  G -> 2,  T -> 3
// < and > are treated separately
// strings start with < are the first 16 values
// strings end with > are the second 16 values
static inline uint8_t hashToken(const char token0, const char token1, const char token2)
{
    // Handle strings starting with '<'
    if (token0 == '<')
    {
        return (char2Val(token1) << 2) + char2Val(token2);
    }
    // Handle strings ending with '>'
    else if (token2 == '>')
    {
        return 16 + (char2Val(token0) << 2) + char2Val(token1);
    }
    // Handle other strings containing 'a', 'c', 'g', 't'
    else
    {
        return 32 + (char2Val(token0) << 4) + (char2Val(token1) << 2) + char2Val(token2);
    }
}

class Preprocessor
{
private:
    std::vector<const char *> tokens_;
    std::vector<uint16_t> indices_;

public:
    Preprocessor()
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

    // extract all substrings of 3 from seq, and return the corresponding index
    inline std::vector<uint16_t> preprocess(const std::string &seq, unsigned maxLen)
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

    std::vector<std::vector<uint16_t>> preprocessBatch(const std::vector<std::string> &seqs, unsigned maxLen)
    {
        std::vector<std::vector<uint16_t>> result(seqs.size());
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
        return result;
    }
};
