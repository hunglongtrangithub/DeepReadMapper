#pragma once

#include "preprocess.hpp"
#include "fast_model.hpp"
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
        const std::string &model_path = "model/finetuned_sgn33-new-a-Apr6.xml",
        size_t batch_size = 100,
        size_t max_len = 123,
        size_t model_out_size = 128);

    /**
     * Vectorizes a batch of input sequences.
     * @param input Vector of input sequences as strings.
     * @return 2D vector of floats representing the vectorized sequences.
     */
    std::vector<std::vector<float>> vectorize(const std::vector<std::string> &input);

private:
    std::vector<std::vector<float>> inference(const std::vector<std::vector<uint16_t>> &batch_input);
    std::vector<std::vector<uint16_t>> transpose_batch(const std::vector<std::vector<uint16_t>> &batch_input);
    std::vector<int64_t> cast_to_int64(const std::vector<std::vector<uint16_t>> &batch_input);

    // Members
    size_t batch_size_;
    size_t max_len_;
    size_t model_out_size_;
    Preprocessor preprocessor_;
    FastModel model_;
};