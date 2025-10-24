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
    // Prepare data buffers for concurrent inference
    size_t max_concurrent = Config::Inference::NUM_INFER_REQUESTS;
    data_buffers_.resize(max_concurrent);

    for (auto &buffer : data_buffers_)
    {
        buffer.resize(batch_size_ * max_len_);
    }
}

/**
 * @brief Wrapper method for the entire vectorization process
 * @details Handles the complete sequence vectorization pipeline including:
 *          - preprocessing
 *          - inference
 * @param input Vector of string sequences to be vectorized
 * @param verbose Whether to print detailed logs (default: false)
 * @return Vector of vectorized sequences as floating-point arrays
 */
std::vector<std::vector<float>> Vectorizer::vectorize(const std::vector<std::string> &input, bool verbose)
{
    if (verbose) {
        std::cout << "[Inference] Vectorizing " << input.size() << " sequences..." << std::endl;
    }

    // [STEP 1] Preprocess the input sequences
    if (verbose) {
        std::cout << "[Inference] Preprocessing input for vectorization..." << std::endl;
    }
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<uint16_t>> preprocessed_inputs = preprocessor_.preprocessBatch(input, max_len_, verbose);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (verbose) {
        std::cout << "[Inference] Preprocessing completed in " << duration << " ms." << std::endl;
    }

    // [STEP 2] Initialize output array
    std::vector<std::vector<float>> output(input.size(), std::vector<float>(model_out_size_));

    // [STEP 3] Inference in batches
    if (verbose) {
        std::cout << "[Inference] Running inference..." << std::endl;
    }

    // Progress bar only shown if verbose
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"vectorizing"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    if (verbose) {
        // Hide cursor and show progress bar
        indicators::show_console_cursor(false);
    }

    size_t total_sequences = input.size();

    // Inference Method 2: Multi-batch processing
    const size_t concurrent_batches = Config::Inference::NUM_INFER_REQUESTS;

    start = std::chrono::high_resolution_clock::now();

    for (size_t start_row = 0; start_row < total_sequences; start_row += batch_size_ * concurrent_batches)
    {
        // Prepare multiple batches for concurrent processing
        std::vector<std::vector<std::vector<uint16_t>>> batches;
        std::vector<size_t> batch_start_indices;

        for (size_t batch_idx = 0; batch_idx < concurrent_batches; ++batch_idx)
        {
            size_t row = start_row + batch_idx * batch_size_;
            if (row >= total_sequences)
                break;

            size_t batch_end = std::min(row + batch_size_, total_sequences);
            size_t current_batch_size = batch_end - row;

            std::vector<std::vector<uint16_t>> batch_input(current_batch_size);
            for (size_t i = 0; i < current_batch_size; ++i)
            {
                batch_input[i] = preprocessed_inputs[row + i];
            }

            batches.push_back(batch_input);
            batch_start_indices.push_back(row);
        }

        // Process all batches concurrently
        std::vector<std::vector<float>> all_results = inferenceBatch(batches);

        // Copy results to output array
        size_t result_idx = 0;
        for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx)
        {
            size_t start_idx = batch_start_indices[batch_idx];
            for (size_t i = 0; i < batches[batch_idx].size(); ++i)
            {
                output[start_idx + i] = all_results[result_idx++];
            }
        }

        // Update progress bar only if verbose
        if (verbose) {
            size_t processed = std::min(start_row + batch_size_ * concurrent_batches, total_sequences);
            size_t current_progress_percent = (processed * 100) / total_sequences;
            progressBar.set_progress(current_progress_percent);
        }
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (verbose) {
        std::cout << "[Inference] Kernel completed in " << duration << " ms." << std::endl;
    }

    // Complete progress bar and show cursor only if verbose
    if (verbose) {
        progressBar.set_progress(100);
        indicators::show_console_cursor(true);
        std::cout << "[Inference] Vectorization completed successfully!" << std::endl;
    }

    return output;
}

/**
 * @brief Performs model inference on a single batch of preprocessed sequences
 * @details This method passes the preprocessed input to the finetuned model for inference.
 *          The process includes padding, transposing, casting, and reshaping operations.
 * @param batch_input Vector of preprocessed sequences to perform inference on
 * @return Vector of vectorized representations for each input sequence
 */
std::vector<std::vector<float>> Vectorizer::inference(const std::vector<std::vector<uint16_t>> &batch_input)
{
    // Add timer for each step to debug performance

    // auto start = std::chrono::high_resolution_clock::now();

    // Fill batch with 0s if batch_input is smaller than batch_size_
    std::vector<std::vector<uint16_t>> padded_batch_input = batch_input;
    size_t batch_size = batch_input.size();

    while (padded_batch_input.size() < batch_size_)
    {
        padded_batch_input.push_back(std::vector<uint16_t>(max_len_, 0));
    }

    // auto end = std::chrono::high_resolution_clock::now();

    // auto pad_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // start = std::chrono::high_resolution_clock::now();
    // Transpose batch
    std::vector<std::vector<uint16_t>> transposed_batch = transpose(padded_batch_input);

    // end = std::chrono::high_resolution_clock::now();

    // auto transpose_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // start = std::chrono::high_resolution_clock::now();
    // Cast to int64_t for model input
    std::vector<int64_t> input_data = castToInt64(transposed_batch);

    // end = std::chrono::high_resolution_clock::now();
    // auto cast_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // start = std::chrono::high_resolution_clock::now();

    // Model inference (synchronous)
    // std::vector<float> model_output = model_(input_data, {max_len_, batch_size_});

    // Model inference (asynchronous)
    auto future_result = model_.inferAsync(input_data, {max_len_, batch_size_});
    std::vector<float> model_output = future_result.get();

    // end = std::chrono::high_resolution_clock::now();
    // auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // start = std::chrono::high_resolution_clock::now();
    // Reshape output into 2D batch format
    std::vector<std::vector<float>> batch_output(batch_size, std::vector<float>(model_out_size_));

    for (size_t i = 0; i < batch_size; ++i)
    {
        for (size_t j = 0; j < model_out_size_; ++j)
        {
            batch_output[i][j] = model_output[i * model_out_size_ + j];
        }
    }
    // end = std::chrono::high_resolution_clock::now();
    // auto reshape_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // std::cout << "Process time per batch: " << std::endl;
    // Convert time to microsec/query for each step
    // pad_time = (pad_time * 1000.0) / batch_size;
    // transpose_time = (transpose_time * 1000.0) / batch_size;
    // cast_time = (cast_time * 1000.0) / batch_size;
    // inference_time = (inference_time * 1000.0) / batch_size;
    // reshape_time = (reshape_time * 1000.0) / batch_size;

    // std::cout << "Padding time: " << pad_time << " microsec/query" << std::endl;
    // std::cout << "Transposing time: " << transpose_time << " microsec/query" << std::endl;
    // std::cout << "Casting time: " << cast_time << " microsec/query" << std::endl;
    // std::cout << "Inference time: " << inference_time << " microsec/query" << std::endl;
    // std::cout << "Reshaping time: " << reshape_time << " microsec/query" << std::endl;

    return batch_output;
}

/**
 * @brief Performs concurrent model inference on multiple batches
 * @details Handles inference for multiple batches concurrently to improve throughput.
 *          The process includes padding, transposing, casting for each batch,
 *          followed by asynchronous inference and result collection.
 * @param batches Vector of batches, where each batch is a vector of preprocessed sequences
 * @return Vector of all vectorized sequences from all batches
 */
std::vector<std::vector<float>> Vectorizer::inferenceBatch(const std::vector<std::vector<std::vector<uint16_t>>> &batches)
{
    std::vector<const std::vector<int64_t> *> batch_input_ptrs;
    std::vector<size_t> original_batch_sizes;

    batch_input_ptrs.reserve(batches.size());
    original_batch_sizes.reserve(batches.size());

    // Prepare all batches using dedicated buffers
    for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx)
    {
        const auto &batch = batches[batch_idx];
        auto &current_buffer = data_buffers_[batch_idx];

        original_batch_sizes.push_back(batch.size());

        prepareBatch(batch, current_buffer);

        // Pass pointer to avoid copy
        batch_input_ptrs.push_back(&current_buffer);
    }

    std::vector<const int64_t *> batch_ptrs;
    batch_ptrs.reserve(batch_input_ptrs.size());
    for (const auto *ptr : batch_input_ptrs)
    {
        batch_ptrs.push_back(ptr->data());
    }

    // Process all batches concurrently
    auto futures = model_.inferBatchAsync(batch_ptrs, {max_len_, batch_size_});

    // Collect results with optimized reshaping
    std::vector<std::vector<float>> all_results;
    all_results.reserve(std::accumulate(original_batch_sizes.begin(), original_batch_sizes.end(), 0));

    for (size_t batch_idx = 0; batch_idx < futures.size(); ++batch_idx)
    {
        std::vector<float> model_output = futures[batch_idx].get();
        size_t original_size = original_batch_sizes[batch_idx];

        // Optimized result extraction using iterators
        for (size_t i = 0; i < original_size; ++i)
        {
            all_results.emplace_back(
                model_output.begin() + i * model_out_size_,
                model_output.begin() + (i + 1) * model_out_size_);
        }
    }

    return all_results;
}

/**
 * @brief Helper function to transpose batch input
 * @details Converts from [num_sequences][sequence_length] to [sequence_length][num_sequences] format
 * @param batch_input Batch of sequences to transpose
 * @return Transposed batch
 */
std::vector<std::vector<uint16_t>> Vectorizer::transpose(const std::vector<std::vector<uint16_t>> &batch_input)
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

/**
 * @brief Helper function to cast batch input to int64_t
 * @details Converts uint16_t values to int64_t as required by the model input
 * @param batch_input Batch of sequences to cast
 * @return Flattened vector of int64_t values
 */
std::vector<int64_t> Vectorizer::castToInt64(const std::vector<std::vector<uint16_t>> &batch_input)
{
    size_t total_size = batch_input.size() * batch_input[0].size();
    std::vector<int64_t> casted_data(total_size);

    auto it = casted_data.begin();
    for (const auto &seq : batch_input)
    {
        it = std::transform(seq.begin(), seq.end(), it, [](uint16_t val)
                            { return static_cast<int64_t>(val); });
    }
    return casted_data;
}

/**
 * @brief Optimized single-pass batch preparation using pre-allocated buffer
 * @details Combines transpose, cast, and padding in one pass for maximum efficiency
 * @param batch Input batch of sequences
 * @param buffer Pre-allocated buffer to write results
 * @return 0 on success
 */
int Vectorizer::prepareBatch(const std::vector<std::vector<uint16_t>> &batch, std::vector<int64_t> &buffer)
{
    size_t actual_batch_size = batch.size();

    // Zero only the memory we'll actually use (much faster for long sequences)
    size_t total_elements = batch_size_ * max_len_;
    std::fill(buffer.begin(), buffer.begin() + total_elements, 0);

    // Single pass: transpose, cast, and pad simultaneously
    for (size_t seq_idx = 0; seq_idx < actual_batch_size; ++seq_idx)
    {
        const auto &sequence = batch[seq_idx];
        size_t seq_len = std::min(sequence.size(), max_len_);

        for (size_t pos = 0; pos < seq_len; ++pos)
        {
            buffer[pos * batch_size_ + seq_idx] = static_cast<int64_t>(sequence[pos]);
        }
    }

    return 0;
}


// ============================================================================
// LongSeqVectorizer Implementation
// ============================================================================

LongSeqVectorizer::LongSeqVectorizer(
    const std::string &model_path,
    size_t chunk_size,
    size_t guard,
    size_t right_ctx,
    float lambda_gc,
    float lambda_pal,
    float lambda_3)
    : chunk_size_(chunk_size),
      guard_(guard),
      right_ctx_(right_ctx),
      lambda_gc_(lambda_gc),
      lambda_pal_(lambda_pal),
      lambda_3_(lambda_3),
      preprocessor_(),
      model_(model_path)
{
    // Prepare data buffers for concurrent inference (same as Vectorizer)
    size_t max_concurrent = Config::Inference::NUM_INFER_REQUESTS;
    size_t batch_size = Config::Inference::BATCH_SIZE;
    data_buffers_.resize(max_concurrent);

    for (auto &buffer : data_buffers_)
    {
        buffer.resize(batch_size * chunk_size_);
    }
}

std::vector<std::vector<float>> LongSeqVectorizer::vectorize(const std::vector<std::string> &seqs, bool verbose)
{
    if (verbose) {
        std::cout << "[LongSeq Vectorizer] Vectorizing " << seqs.size() << " long sequences..." << std::endl;
    }

    auto start_total = std::chrono::high_resolution_clock::now();

    // [STEP 1] Chunk all sequences and flatten into chunk list
    if (verbose) {
        std::cout << "[LongSeq Vectorizer] Chunking sequences..." << std::endl;
    }

    struct ChunkInfo {
        std::string chunk_text;
        size_t seq_idx;
        size_t chunk_idx;
        size_t total_chunks_for_seq;
    };

    std::vector<ChunkInfo> all_chunks;
    std::vector<size_t> seq_chunk_counts(seqs.size(), 0);

    for (size_t seq_idx = 0; seq_idx < seqs.size(); ++seq_idx)
    {
        size_t phase_offset = 0;
        auto chunks = chunkSeq(seqs[seq_idx], phase_offset);
        seq_chunk_counts[seq_idx] = chunks.size();

        for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx)
        {
            all_chunks.push_back({chunks[chunk_idx], seq_idx, chunk_idx, chunks.size()});
        }
    }

    if (verbose) {
        std::cout << "[LongSeq Vectorizer] Generated " << all_chunks.size() 
                  << " chunks from " << seqs.size() << " sequences." << std::endl;
    }

    // [STEP 2] Preprocess all chunks
    if (verbose) {
        std::cout << "[LongSeq Vectorizer] Preprocessing chunks..." << std::endl;
    }

    std::vector<std::vector<uint16_t>> all_tokens;
    all_tokens.reserve(all_chunks.size());

    for (const auto &chunk_info : all_chunks)
    {
        auto tokens = preprocessor_.preprocess(chunk_info.chunk_text, chunk_size_);
        all_tokens.push_back(tokens);
    }

    // [STEP 3] Concurrent batch inference (using same pattern as Vectorizer)
    if (verbose) {
        std::cout << "[LongSeq Vectorizer] Running concurrent inference..." << std::endl;
    }

    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"long vectorizing"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    if (verbose) {
        indicators::show_console_cursor(false);
    }

    const size_t batch_size = Config::Inference::BATCH_SIZE;
    const size_t concurrent_batches = Config::Inference::NUM_INFER_REQUESTS;
    size_t total_chunks = all_tokens.size();

    std::vector<std::vector<float>> chunk_embeddings(total_chunks);

    auto start_infer = std::chrono::high_resolution_clock::now();

    for (size_t start_idx = 0; start_idx < total_chunks; start_idx += batch_size * concurrent_batches)
    {
        // Prepare multiple batches for concurrent processing
        std::vector<std::vector<std::vector<uint16_t>>> batches;
        std::vector<size_t> batch_start_indices;

        for (size_t batch_idx = 0; batch_idx < concurrent_batches; ++batch_idx)
        {
            size_t chunk_idx = start_idx + batch_idx * batch_size;
            if (chunk_idx >= total_chunks)
                break;

            size_t batch_end = std::min(chunk_idx + batch_size, total_chunks);
            size_t current_batch_size = batch_end - chunk_idx;

            std::vector<std::vector<uint16_t>> batch_input(current_batch_size);
            for (size_t i = 0; i < current_batch_size; ++i)
            {
                batch_input[i] = all_tokens[chunk_idx + i];
            }

            batches.push_back(batch_input);
            batch_start_indices.push_back(chunk_idx);
        }

        // Process all batches concurrently using inferBatchAsync
        std::vector<const int64_t *> batch_ptrs;
        std::vector<size_t> original_batch_sizes;
        batch_ptrs.reserve(batches.size());
        original_batch_sizes.reserve(batches.size());

        // Prepare all batches using dedicated buffers
        for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx)
        {
            const auto &batch = batches[batch_idx];
            auto &current_buffer = data_buffers_[batch_idx];

            original_batch_sizes.push_back(batch.size());
            prepareBatch(batch, current_buffer);
            batch_ptrs.push_back(current_buffer.data());
        }

        // Async inference
        auto futures = model_.inferBatchAsync(batch_ptrs, {chunk_size_, batch_size});

        // Collect results
        for (size_t batch_idx = 0; batch_idx < futures.size(); ++batch_idx)
        {
            std::vector<float> model_output = futures[batch_idx].get();
            size_t original_size = original_batch_sizes[batch_idx];
            size_t start_chunk_idx = batch_start_indices[batch_idx];

            for (size_t i = 0; i < original_size; ++i)
            {
                chunk_embeddings[start_chunk_idx + i] = std::vector<float>(
                    model_output.begin() + i * 128,
                    model_output.begin() + (i + 1) * 128
                );
            }
        }

        // Update progress bar
        if (verbose) {
            size_t processed = std::min(start_idx + batch_size * concurrent_batches, total_chunks);
            size_t current_progress_percent = (processed * 100) / total_chunks;
            progressBar.set_progress(current_progress_percent);
        }
    }

    auto end_infer = std::chrono::high_resolution_clock::now();
    auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_infer - start_infer).count();

    if (verbose) {
        progressBar.set_progress(100);
        indicators::show_console_cursor(true);
        std::cout << "[LongSeq Vectorizer] Inference completed in " << infer_duration << " ms." << std::endl;
    }

    // [STEP 4] Aggregate chunks back to sequences
    if (verbose) {
        std::cout << "[LongSeq Vectorizer] Aggregating chunks..." << std::endl;
    }

    std::vector<std::vector<float>> results(seqs.size());

    size_t chunk_offset = 0;
    for (size_t seq_idx = 0; seq_idx < seqs.size(); ++seq_idx)
    {
        size_t num_chunks = seq_chunk_counts[seq_idx];

        if (num_chunks == 0)
        {
            results[seq_idx] = std::vector<float>(128, 0.0f);
        }
        else if (num_chunks == 1)
        {
            results[seq_idx] = chunk_embeddings[chunk_offset];
        }
        else
        {
            // Extract chunk embeddings for this sequence
            std::vector<std::vector<float>> seq_chunk_embs(
                chunk_embeddings.begin() + chunk_offset,
                chunk_embeddings.begin() + chunk_offset + num_chunks
            );

            // Get original chunks for weight computation
            std::vector<std::string> seq_chunks;
            seq_chunks.reserve(num_chunks);
            for (size_t i = 0; i < num_chunks; ++i)
            {
                seq_chunks.push_back(all_chunks[chunk_offset + i].chunk_text);
            }

            // Compute weights and interpolate
            auto weights = computeMotifWeights(seq_chunks);
            results[seq_idx] = weightedInterpolate(seq_chunk_embs, weights);
        }

        chunk_offset += num_chunks;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();

    if (verbose) {
        std::cout << "[LongSeq Vectorizer] Total vectorization completed in " << total_duration << " ms." << std::endl;
    }

    return results;
}

int LongSeqVectorizer::prepareBatch(const std::vector<std::vector<uint16_t>> &batch, std::vector<int64_t> &buffer)
{
    size_t batch_size = Config::Inference::BATCH_SIZE;
    size_t actual_batch_size = batch.size();

    // Zero the buffer
    size_t total_elements = batch_size * chunk_size_;
    std::fill(buffer.begin(), buffer.begin() + total_elements, 0);

    // Single pass: transpose, cast, and pad simultaneously
    for (size_t seq_idx = 0; seq_idx < actual_batch_size; ++seq_idx)
    {
        const auto &sequence = batch[seq_idx];
        size_t seq_len = std::min(sequence.size(), chunk_size_);

        for (size_t pos = 0; pos < seq_len; ++pos)
        {
            buffer[pos * batch_size + seq_idx] = static_cast<int64_t>(sequence[pos]);
        }
    }

    return 0;
}

// ...existing code for chunkSeq, computeMotifWeights, weightedInterpolate, etc...

std::vector<std::string> LongSeqVectorizer::chunkSeq(const std::string &seq, size_t &phase_offset)
{
    phase_offset = 0;
    size_t delta = phase_offset % 3;
    
    size_t L = seq.length();
    
    if (L < chunk_size_)
    {
        return {seq};
    }
    
    std::vector<std::string> chunks;
    size_t i = delta;
    
    while (i + chunk_size_ <= L)
    {
        chunks.push_back(seq.substr(i, chunk_size_));
        i += chunk_size_;
    }
    
    return chunks;
}

std::vector<float> LongSeqVectorizer::computeMotifWeights(const std::vector<std::string> &chunks)
{
    std::vector<float> weights;
    weights.reserve(chunks.size());
    
    for (const auto &chunk : chunks)
    {
        float gc = computeGC(chunk);
        float pal = computePalindrome(chunk);
        float spec = compute3merSpectrum(chunk);
        
        float w = std::exp(lambda_gc_ * gc + lambda_pal_ * pal + lambda_3_ * spec);
        weights.push_back(w);
    }
    
    return weights;
}

std::vector<float> LongSeqVectorizer::weightedInterpolate(const std::vector<std::vector<float>> &chunk_embs,
                                                          const std::vector<float> &weights)
{
    size_t T = chunk_embs.size();
    size_t d = 128;
    
    if (T == 1) return chunk_embs[0];
    
    std::vector<float> Y(d, 0.0f);
    
    // Interpolate across chunks to get final 128-dim embedding
    for (size_t dim = 0; dim < d; ++dim)
    {
        float weighted_sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (size_t t = 0; t < T; ++t)
        {
            weighted_sum += weights[t] * chunk_embs[t][dim];
            weight_sum += weights[t];
        }
        
        Y[dim] = weighted_sum / weight_sum;
    }
    
    return Y;
}

float LongSeqVectorizer::computeGC(const std::string &chunk)
{
    int gc_count = 0;
    int total = 0;
    
    for (char c : chunk)
    {
        char upper = std::toupper(c);
        if (upper == 'A' || upper == 'T' || upper == 'C' || upper == 'G')
        {
            total++;
            if (upper == 'G' || upper == 'C')
            {
                gc_count++;
            }
        }
    }
    
    return total > 0 ? static_cast<float>(gc_count) / total : 0.0f;
}

float LongSeqVectorizer::computePalindrome(const std::string &chunk)
{
    auto complement = [](char c) -> char {
        switch (std::toupper(c))
        {
        case 'A': return 'T';
        case 'T': return 'A';
        case 'C': return 'G';
        case 'G': return 'C';
        default: return 'N';
        }
    };
    
    int matches = 0;
    int total = 0;
    size_t len = chunk.length();
    
    for (size_t i = 0; i < len / 2; ++i)
    {
        char left = std::toupper(chunk[i]);
        char right = std::toupper(chunk[len - 1 - i]);
        
        if (left != 'N' && right != 'N')
        {
            total++;
            if (left == complement(right))
            {
                matches++;
            }
        }
    }
    
    return total > 0 ? static_cast<float>(matches) / total : 0.0f;
}

float LongSeqVectorizer::compute3merSpectrum(const std::string &chunk)
{
    std::unordered_map<std::string, int> kmer_counts;
    
    for (size_t i = 0; i + 3 <= chunk.length(); ++i)
    {
        std::string kmer = chunk.substr(i, 3);
        for (char &c : kmer)
        {
            c = std::toupper(c);
        }
        
        bool valid = true;
        for (char c : kmer)
        {
            if (c != 'A' && c != 'T' && c != 'C' && c != 'G')
            {
                valid = false;
                break;
            }
        }
        
        if (valid)
        {
            kmer_counts[kmer]++;
        }
    }
    
    float norm = 0.0f;
    for (const auto &pair : kmer_counts)
    {
        norm += pair.second * pair.second;
    }
    
    return std::sqrt(norm);
}