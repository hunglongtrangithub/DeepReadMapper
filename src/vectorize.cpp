#include "vectorize.hpp"
#include "progressbar.h"

Vectorizer::Vectorizer(
    const std::string &model_path,
    size_t batch_size,
    size_t max_len,
    size_t model_out_size)
    : batch_size_(batch_size),
      max_len_(max_len),
      model_out_size_(model_out_size),
      preprocessor_(),
      model_(model_path)
{
}

std::vector<std::vector<float>> Vectorizer::vectorize(const std::vector<std::string> &input)
{
    /*
    Wrapper method, includes all steps of vectorize process:
    - preprocessing
    - inference
    */

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

    // Hide cursor and create progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"vectorizing"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    size_t total_sequences = input.size();

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

        // Run inference on batch
        std::vector<std::vector<float>> batch_output = inference(batch_input);

        // Copy results to output array
        for (size_t i = 0; i < current_batch_size; ++i)
        {
            output[row + i] = batch_output[i];
        }

        // Update progress bar
        size_t current_progress_percent = (batch_end * 100) / total_sequences;
        progressBar.set_progress(current_progress_percent);
    }

    // Complete progress bar and show cursor
    progressBar.set_progress(100);
    indicators::show_console_cursor(true);
    
    std::cout << "Vectorization completed successfully!" << std::endl;
    return output;
}

std::vector<std::vector<float>> Vectorizer::inference(const std::vector<std::vector<uint16_t>> &batch_input)
{
    // This method passes the preprocessed input to the finetuned model for inference.

    // Fill batch with 0s if batch_input is smaller than batch_size_
    std::vector<std::vector<uint16_t>> padded_batch_input = batch_input;
    size_t original_batch_size = batch_input.size();

    while (padded_batch_input.size() < batch_size_)
    {
        padded_batch_input.push_back(std::vector<uint16_t>(max_len_, 0));
    }

    // Transpose batch
    std::vector<std::vector<uint16_t>> transposed_batch = transpose_batch(padded_batch_input);

    // Cast to int64_t for model input
    std::vector<int64_t> input_data = cast_to_int64(transposed_batch);

    // Pass through model with fixed shape
    std::vector<float> model_output = model_(input_data, {max_len_, batch_size_});

    // Reshape output into 2D batch format
    std::vector<std::vector<float>> batch_output(original_batch_size, std::vector<float>(model_out_size_));

    for (size_t i = 0; i < original_batch_size; ++i)
    {
        for (size_t j = 0; j < model_out_size_; ++j)
        {
            batch_output[i][j] = model_output[i * model_out_size_ + j];
        }
    }

    return batch_output;
}

std::vector<std::vector<uint16_t>> Vectorizer::transpose_batch(const std::vector<std::vector<uint16_t>> &batch_input)
{
    // Helper function to transpose batch input

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

std::vector<int64_t> Vectorizer::cast_to_int64(const std::vector<std::vector<uint16_t>> &batch_input)
{
    // Helper function to cast batch input to int64_t
    std::vector<int64_t> casted_data;

    // Precalculate total size to avoid memory reallocation
    size_t total_size = 0;
    for (const auto &seq : batch_input)
    {
        total_size += seq.size();
    }
    casted_data.reserve(total_size);

    for (const auto &seq : batch_input)
    {
        casted_data.insert(casted_data.end(), seq.begin(), seq.end());
    }
    return casted_data;
}