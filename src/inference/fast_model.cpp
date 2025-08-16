#include "fast_model.hpp"

FastModel::FastModel(const std::string &model_path) : model_path(model_path)
{
    try
    {
        // Use Multi-threaded to parallel OpenVino layers (pipelining)
        ov::AnyMap config = {
            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
            ov::inference_num_threads(Config::Inference::NUM_THREADS),
            //  ov::streams::num(Config::Inference::NUM_STREAMS)
        };

        // Load the network in Inference Engine
        auto model = core.read_model(model_path);
        compiled_model = core.compile_model(model, "CPU", config);
        // compiled_model = core.compile_model(model, "CPU");

        // Get output layer
        output_layer = compiled_model.output(0);

        std::cout << "Model loaded successfully: " << model_path << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        throw;
    }
}

/*
Method to run inference on batched preprocessed sequences.
*/
std::vector<float> FastModel::operator()(const std::vector<int64_t> &input_data, const std::vector<size_t> &input_shape)
{
    try
    {
        // Create input tensor
        auto input_port = compiled_model.input(0);
        ov::Tensor input_tensor(input_port.get_element_type(), input_shape);

        // Copy input data to tensor
        int64_t *tensor_data = input_tensor.data<int64_t>();
        std::memcpy(tensor_data, input_data.data(), input_data.size() * sizeof(int64_t));

        // Create inference request
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        infer_request.set_input_tensor(input_tensor);

        // Run inference (synchronous)
        infer_request.infer();

        // Get output tensor - use index 0 instead of output_layer
        auto output_tensor = infer_request.get_output_tensor(0);

        // Convert output to vector
        const float *output_data = output_tensor.data<float>();
        size_t output_size = output_tensor.get_size();

        // Return raw output (flattened)
        return std::vector<float>(output_data, output_data + output_size);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        throw;
    }
}

std::future<std::vector<float>> FastModel::inferAsync(const std::vector<int64_t> &input_data, const std::vector<size_t> &input_shape)
{
    /*
    This method performs asynchronous inference on the model using OpenVINO's async API properly.
    */
    return std::async(std::launch::async, [this, input_data, input_shape]() -> std::vector<float>
                      {
        try {
            // Create input tensor
            auto input_port = compiled_model.input(0);
            ov::Tensor input_tensor(input_port.get_element_type(), input_shape);

            int64_t *tensor_data = input_tensor.data<int64_t>();
            std::memcpy(tensor_data, input_data.data(), input_data.size() * sizeof(int64_t));

            // Create an inference request
            ov::InferRequest infer_request = compiled_model.create_infer_request();
            infer_request.set_input_tensor(input_tensor);

            // Use synchronous inference within this async thread
            infer_request.infer();

            // Get the output tensor and convert it to a vector of floats
            auto output_tensor = infer_request.get_output_tensor(0);
            const float *output_data = output_tensor.data<float>();
            size_t output_size = output_tensor.get_size();

            return std::vector<float>(output_data, output_data + output_size);
        } catch (const std::exception &e) {
            std::cerr << "Error during asynchronous inference: " << e.what() << std::endl;
            throw;
        } });
}

/*
This method uses OpenVINO's async API properly:
1. Create all inference requests and tensors
2. Start all async operations
3. Return futures that will wait for completion when needed
*/
std::vector<std::future<std::vector<float>>> FastModel::inferBatchAsync(const std::vector<std::vector<int64_t>> &batch_inputs, const std::vector<size_t> &batch_shape)
{

    // Use shared_ptr to keep the data alive for the lifetime of the futures
    auto inference_data = std::make_shared<std::vector<AsyncInferenceData>>();
    std::vector<std::future<std::vector<float>>> futures;

    inference_data->reserve(batch_inputs.size());
    futures.reserve(batch_inputs.size());

    // Phase 1: Create all inference requests and tensors
    for (size_t i = 0; i < batch_inputs.size(); ++i)
    {
        AsyncInferenceData data;

        // Create input tensor
        auto input_port = compiled_model.input(0);
        data.input_tensor = ov::Tensor(input_port.get_element_type(), batch_shape);

        // Copy input data (we need to keep it alive)
        data.input_data_copy = batch_inputs[i];
        int64_t *tensor_data = data.input_tensor.data<int64_t>();
        std::memcpy(tensor_data, data.input_data_copy.data(), data.input_data_copy.size() * sizeof(int64_t));

        // Create inference request
        data.request = compiled_model.create_infer_request();
        data.request.set_input_tensor(data.input_tensor);

        inference_data->push_back(std::move(data));
    }

    // Phase 2: Start all async operations
    for (auto &data : *inference_data)
    {
        data.request.start_async();
    }

    // Phase 3: Create futures that will wait for completion
    for (size_t i = 0; i < inference_data->size(); ++i)
    {
        futures.push_back(std::async(std::launch::deferred, [inference_data, i]() -> std::vector<float>
                                     {
            try {
                // Wait for this specific request to complete
                (*inference_data)[i].request.wait();
                
                // Get the output tensor and convert it to a vector of floats
                auto output_tensor = (*inference_data)[i].request.get_output_tensor(0);
                const float *output_data = output_tensor.data<float>();
                size_t output_size = output_tensor.get_size();
                
                return std::vector<float>(output_data, output_data + output_size);
            } catch (const std::exception &e) {
                std::cerr << "Error during batch async inference: " << e.what() << std::endl;
                throw;
            } }));
    }

    return futures;
}

/*
A method to perform asynchronous inference for a batch of requests using pointers.
*/
std::vector<std::future<std::vector<float>>> FastModel::inferBatchAsync(const std::vector<const int64_t *> &batch_ptrs, const std::vector<size_t> &batch_shape)
{
    auto inference_data = std::make_shared<std::vector<AsyncInferenceData>>();
    std::vector<std::future<std::vector<float>>> futures;

    inference_data->reserve(batch_ptrs.size());
    futures.reserve(batch_ptrs.size());

    // Phase 1: Create all inference requests and tensors
    for (size_t i = 0; i < batch_ptrs.size(); ++i)
    {
        AsyncInferenceData data;

        auto input_port = compiled_model.input(0);
        data.input_tensor = ov::Tensor(input_port.get_element_type(), batch_shape);

        // Direct memcpy from pointer (no vector copy needed)
        int64_t *tensor_data = data.input_tensor.data<int64_t>();
        size_t data_size = batch_shape[0] * batch_shape[1];
        std::memcpy(tensor_data, batch_ptrs[i], data_size * sizeof(int64_t));

        data.request = compiled_model.create_infer_request();
        data.request.set_input_tensor(data.input_tensor);

        inference_data->push_back(std::move(data));
    }

    // Phase 2: Start all async operations
    for (auto &data : *inference_data)
    {
        data.request.start_async();
    }

    // Phase 3: Create futures that will wait for completion
    for (size_t i = 0; i < inference_data->size(); ++i)
    {
        futures.push_back(std::async(std::launch::deferred, [inference_data, i]() -> std::vector<float>
                                     {
            try {
                (*inference_data)[i].request.wait();
                auto output_tensor = (*inference_data)[i].request.get_output_tensor(0);
                const float *output_data = output_tensor.data<float>();
                size_t output_size = output_tensor.get_size();
                return std::vector<float>(output_data, output_data + output_size);
            } catch (const std::exception &e) {
                std::cerr << "Error during batch async inference: " << e.what() << std::endl;
                throw;
            } }));
    }

    return futures;
}