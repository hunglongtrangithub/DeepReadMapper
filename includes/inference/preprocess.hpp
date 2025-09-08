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
    Preprocessor();
    
    std::vector<uint16_t> preprocess(const std::string &seq, unsigned maxLen);
    
    std::vector<std::vector<uint16_t>> preprocessBatch(const std::vector<std::string> &seqs, unsigned maxLen, bool verbose = true);
};
