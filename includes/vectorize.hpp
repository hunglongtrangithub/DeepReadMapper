#pragma once

#include "preprocess.hpp"
#include "fast_model.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

class Vectorizer
{
public:
    Vectorizer(
        const std::string &model_path = "model/finetuned_sgn33-new-a-Apr6.xml",
        size_t batch_size = 100,
        size_t max_len = 123,
        size_t model_out_size = 128);

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