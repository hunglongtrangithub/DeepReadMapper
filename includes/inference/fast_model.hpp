#pragma once

#include <openvino/openvino.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <cstring>

class FastModel
{
private:
    std::string model_path;
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::Output<const ov::Node> output_layer;

public:
    FastModel(const std::string &model_path = "model/finetuned_sgn33-new-a-Apr6.xml");
    std::vector<float> operator()(const std::vector<int64_t> &input_data, const std::vector<size_t> &input_shape);
};