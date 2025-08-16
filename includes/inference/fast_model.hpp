#pragma once

#include "config.hpp"
#include <openvino/openvino.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <cstring>
#include <future>
#include <memory>

/// @brief FastModel class for handling OpenVINO model inference.
class FastModel
{
private:
    std::string model_path;
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::Output<const ov::Node> output_layer;

    /**
     * @brief Internal structure to hold asynchronous inference data.
     * Used to keep inference requests and associated data alive during async operations.
     */
    struct AsyncInferenceData
    {
        ov::InferRequest request;
        ov::Tensor input_tensor;
        std::vector<int64_t> input_data_copy;
    };

public:
    /**
     * @brief Constructor for FastModel.
     * @param model_path The path to the OpenVINO model XML file.
     * Defaults to "model/finetuned_sgn33-new-a-Apr6.xml".
     */
    FastModel(const std::string &model_path = "model/finetuned_sgn33-new-a-Apr6.xml");

    /**
     * @brief Performs synchronous inference on the model.
     * @param input_data A flat vector of input data for the model.
     * @param input_shape A vector representing the shape of the input data.
     * @return A vector of float representing the inference results.
     */
    std::vector<float> operator()(const std::vector<int64_t> &input_data,
                                  const std::vector<size_t> &input_shape);

    /**
     * @brief Performs asynchronous inference on the model.
     * @param input_data A flat vector of input data for the model.
     * @param input_shape A vector representing the shape of the input data.
     * @return A future object that will eventually hold the inference results.
     */
    std::future<std::vector<float>>
    inferAsync(const std::vector<int64_t> &input_data,
               const std::vector<size_t> &input_shape);

    /**
     * @brief Performs asynchronous inference for a batch of requests.
     * @param batch_inputs A vector of input data vectors for each request in the batch.
     * @param batch_shape An input shape vectors for each request in the batch.
     * @return A vector of future objects, each holding the inference results for a single request.
     */
    std::vector<std::future<std::vector<float>>> inferBatchAsync(const std::vector<std::vector<int64_t>> &batch_inputs, const std::vector<size_t> &batch_shape);

    /**
     * @brief Performs asynchronous inference for a batch of requests using pointers.
     * @param batch_ptrs A vector of pointers to input data for each request in the batch.
     * @param batch_shape An input shape vectors for each request in the batch.
     * @return A vector of future objects, each holding the inference results for a single request
     */

    std::vector<std::future<std::vector<float>>> inferBatchAsync(const std::vector<const int64_t *> &batch_ptrs, const std::vector<size_t> &batch_shape);
};