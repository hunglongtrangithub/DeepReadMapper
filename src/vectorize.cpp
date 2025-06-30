#include "preprocess.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
// #include <torch/torch.h>
// #include <torch/script.h>

class Vectorizer
{
    // This class is responsible for vectorizing input sequences. Process: Apply preprocessor to inputs (tokenize), then inference through finetuned model.
public:
    Vectorizer(
        const std::string &model_path = "model/finetuned_sgn33-new-a-Apr6.xml",
        size_t batch_size = 100, size_t max_len = 123, size_t model_out_size = 128)
        : model_path_(model_path),
          batch_size_(batch_size),
          max_len_(max_len),
          model_out_size_(model_out_size),
          preprocessor_()
    {
        // Constructor implementation

        // TODO: load model in torch
        // torch::jit::script::Module model;
        // try
        // {
        //     model = torch::jit::load("model/model_traced.pt");
        // }
        // catch (const c10::Error &e)
        // {
        //     std::cerr << "Error loading the model:\n";
        //     return -1;
        // }
    }

    std::vector<std::vector<float>> vectorize(const std::vector<std::string> &input)
    {
        // A master method that includes all steps of the vectorization process: preprocessing, inference, and displaying the vectorized output.

        std::cout << "Starting vectorization of " << input.size() << " sequences..." << std::endl;

        // [STEP 1] Preprocess the input sequences
        std::cout << "Preprocessing input for vectorization..." << std::endl;
        std::vector<std::vector<uint16_t>> preprocessed_inputs = preprocessor_.preprocessBatch(input, max_len_);

        // Truncate sequences to max_len_
        for (auto &seq : preprocessed_inputs)
        {
            if (seq.size() > max_len_)
            {
                seq.resize(max_len_);
            }
        }

        // [STEP 2] Initialize output array
        std::vector<std::vector<float>> output(input.size(), std::vector<float>(model_out_size_));

        // [STEP 3] Inference in batches
        std::cout << "Running inference..." << std::endl;

        size_t total_sequences = input.size();
        size_t last_progress_percent = 0;

        for (size_t row = 0; row < total_sequences; row += batch_size_)
        {
            size_t batch_end = std::min(row + batch_size_, total_sequences);
            size_t current_batch_size = batch_end - row;

            // Create batch input
            std::vector<std::vector<uint16_t>> batch_input(current_batch_size);
            for (size_t i = 0; i < current_batch_size; ++i)
            {
                batch_input[i] = preprocessed_inputs[row + i];
            }

            // Transpose batch input (equivalent to Python: batch_input.transpose(1, 0))
            std::vector<std::vector<uint16_t>> transposed_batch = transpose_batch(batch_input);

            // Run inference
            std::vector<std::vector<float>> batch_output = inference(transposed_batch);

            // Copy results to output array
            for (size_t i = 0; i < current_batch_size; ++i)
            {
                output[row + i] = batch_output[i];
            }

            // Print progress every 5%
            size_t current_progress_percent = (batch_end * 100) / total_sequences;
            if (current_progress_percent >= last_progress_percent + 5 || batch_end == total_sequences)
            {
                std::cout << "Progress: " << current_progress_percent << "% ("
                          << batch_end << "/" << total_sequences << " sequences)" << std::endl;
                last_progress_percent = current_progress_percent;
            }
        }

        std::cout << "Vectorization completed successfully!" << std::endl;
        return output;
    }

private:
    std::vector<std::vector<float>> inference(const std::vector<std::vector<uint16_t>> &batch_input)
    {
        // This method passes the preprocessed input to the finetuned model for inference.
        // TODO: Implement actual inference logic using the model at model_path_.

        // For now, return dummy data with correct shape with all zeros.
        std::vector<std::vector<float>> dummy_output(batch_input[0].size(), std::vector<float>(model_out_size_, 0.0f));
        return dummy_output;
    }

    // Helper function to transpose batch input
    std::vector<std::vector<uint16_t>> transpose_batch(const std::vector<std::vector<uint16_t>> &batch_input)
    {
        if (batch_input.empty() || batch_input[0].empty())
        {
            return {};
        }

        size_t num_sequences = batch_input.size();
        size_t sequence_length = batch_input[0].size();

        // Create transposed matrix: [sequence_length][num_sequences]
        std::vector<std::vector<uint16_t>> transposed(sequence_length, std::vector<uint16_t>(num_sequences));

        for (size_t i = 0; i < num_sequences; ++i)
        {
            for (size_t j = 0; j < std::min(sequence_length, batch_input[i].size()); ++j)
            {
                transposed[j][i] = batch_input[i][j];
            }
        }

        return transposed;
    }

    // Members
    std::string model_path_;
    size_t batch_size_;
    size_t max_len_;
    size_t model_out_size_;
    Preprocessor preprocessor_;
};