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
     * @param verbose Whether to print detailed logs (default: false)
     * @return 2D vector of floats representing the vectorized sequences.
     */
    std::vector<std::vector<float>> vectorize(const std::vector<std::string> &input, bool verbose = true);

private:
    std::vector<std::vector<float>> inferenceBatch(const std::vector<std::vector<std::vector<uint16_t>>> &batches);
    int prepareBatch(const std::vector<std::vector<uint16_t>> &batch, std::vector<int64_t> &buffer);

    // These functions are not recommended for performance usage. Use inferenceBatch & prepareBatch instead.
    std::vector<std::vector<float>> inference(const std::vector<std::vector<uint16_t>> &batch_input);
    std::vector<std::vector<uint16_t>> transpose(const std::vector<std::vector<uint16_t>> &batch_input);
    std::vector<int64_t> castToInt64(const std::vector<std::vector<uint16_t>> &batch_input);

    // Members
    size_t batch_size_; // Maximum number of sequences per batch. Actual batch may be smaller.
    size_t max_len_;    // Sequence length
    size_t model_out_size_;
    Preprocessor preprocessor_;
    FastModel model_;

    std::vector<std::vector<int64_t>> data_buffers_; // Unified buffers handled concurrently
};


/// @brief A modern approach to consider all parts of long sequences.
/// @details Splits long sequences into overlapping chunks, infers each chunk,
///          pools codon phases, computes motif-based weights, and combines chunk embeddings.
class LongSeqVectorizer {
private:
  // Members in LongSeqVectorizer
  size_t chunk_size_;
  size_t guard_;
  size_t right_ctx_;
  float lambda_gc_, lambda_pal_, lambda_3_;
  Preprocessor preprocessor_;
  FastModel model_;
  std::vector<std::vector<int64_t>> data_buffers_;
  
  // Helper methods
  std::vector<std::string> chunkSeq(const std::string& seq, size_t& phase_offset);
  int prepareBatch(const std::vector<std::vector<uint16_t>> &batch, std::vector<int64_t> &buffer);
  std::vector<float> computeMotifWeights(const std::vector<std::string>& chunks);
  std::vector<float> weightedInterpolate(const std::vector<std::vector<float>>& chunk_embs, 
                                         const std::vector<float>& weights);
  
  float computeGC(const std::string& chunk);
  float computePalindrome(const std::string& chunk);
  float compute3merSpectrum(const std::string& chunk);
  
public:
  LongSeqVectorizer(
    const std::string &model_path = Config::Inference::MODEL_PATH,
    size_t chunk_size = Config::Inference::MAX_LEN,
    size_t guard = 2,
    size_t right_ctx = 30,
    float lambda_gc = 0.0f,
    float lambda_pal = 0.0f,
    float lambda_3 = 0.0f
  );
  
  std::vector<std::vector<float>> vectorize(const std::vector<std::string>& seqs, bool verbose = true);
};