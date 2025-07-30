#pragma once

#include "progressbar.h"
#include "preprocess.hpp"
#include "fast_model.hpp"
#include "utils.hpp"
#include "config.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

/// @brief class for transforming sequences into suitable vector representations.
/// @details Handles preprocessing and model inference in batches.
class Vectorizer
{
public:
    /**
     * Constructor for Vectorizer.
     * @param model_path Path to the model file.
     * @param batch_size Number of sequences per batch.
     * @param max_len Maximum sequence length.
     * @param model_out_size Output vector size from the model.
     */
    Vectorizer(
        const std::string &model_path = Config::Inference::MODEL_PATH,
        size_t batch_size = Config::Inference::BATCH_SIZE,
        size_t max_len = Config::Inference::MAX_LEN,
        size_t model_out_size = Config::Inference::MODEL_OUT_SIZE);

    /**
     * Vectorizes a batch of input sequences.
     * @param input Vector of input sequences as strings.
     * @return 2D vector of floats representing the vectorized sequences.
     */
    std::vector<std::vector<float>> vectorize(const std::vector<std::string> &input);

private:
    std::vector<std::vector<float>> inference(const std::vector<std::vector<uint16_t>> &batch_input);
    std::vector<std::vector<float>> inferenceBatch(const std::vector<std::vector<std::vector<uint16_t>>> &batches);
    std::vector<std::vector<uint16_t>> transpose(const std::vector<std::vector<uint16_t>> &batch_input);
    std::vector<int64_t> castToInt64(const std::vector<std::vector<uint16_t>> &batch_input);

    // Members
    size_t batch_size_; // Maximum number of sequences per batch. Actual batch may be smaller.
    size_t max_len_;    // Sequence length
    size_t model_out_size_;
    Preprocessor preprocessor_;
    FastModel model_;
};
