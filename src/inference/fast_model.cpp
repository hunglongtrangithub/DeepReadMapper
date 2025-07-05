#include "fast_model.hpp"

FastModel::FastModel(const std::string &model_path) : model_path(model_path)
{
    try
    {
        // Load the network in Inference Engine
        auto model = core.read_model(model_path);
        compiled_model = core.compile_model(model, "CPU");

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

std::vector<float> FastModel::operator()(const std::vector<int64_t> &input_data, const std::vector<size_t> &input_shape)
{
    /*
    Method to run inference on batched preprocessed sequences.
    */
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

        // Run inference
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