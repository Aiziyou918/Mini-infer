#include "mini_infer/operators/relu.h"
#include <algorithm>
#include <cstring>

namespace mini_infer {
namespace operators {

ReLU::ReLU() : Operator("ReLU") {}

core::Status ReLU::forward(
    const std::vector<std::shared_ptr<core::Tensor>>& inputs,
    std::vector<std::shared_ptr<core::Tensor>>& outputs) {
    
    // Input validation
    if (inputs.size() != 1) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    const auto& input = inputs[0];
    if (!input) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Get input information
    const auto& input_shape = input->shape();
    const auto input_dtype = input->dtype();
    
    // Create output tensor (same shape and type as input)
    auto output = core::Tensor::create(input_shape, input_dtype);
    if (!output) {
        return core::Status::ERROR_OUT_OF_MEMORY;
    }
    
    // Calculate total number of elements
    size_t total_elements = static_cast<size_t>(input_shape.numel());
    
    // Execute ReLU calculation based on data type
    if (input_dtype == core::DataType::FLOAT32) {
        const float* input_data = static_cast<const float*>(input->data());
        float* output_data = static_cast<float*>(output->data());
        
        for (size_t i = 0; i < total_elements; ++i) {
            output_data[i] = std::max(0.0f, input_data[i]);
        }
    } else if (input_dtype == core::DataType::INT32) {
        const int32_t* input_data = static_cast<const int32_t*>(input->data());
        int32_t* output_data = static_cast<int32_t*>(output->data());
        
        for (size_t i = 0; i < total_elements; ++i) {
            output_data[i] = std::max(0, input_data[i]);
        }
    } else {
        // Unsupported data type
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Set output
    outputs.clear();
    outputs.push_back(output);
    
    return core::Status::SUCCESS;
}

core::Status ReLU::infer_shape(
    const std::vector<core::Shape>& input_shapes,
    std::vector<core::Shape>& output_shapes) {
    
    // Input validation
    if (input_shapes.size() != 1) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // ReLU does not change the shape of the tensor
    output_shapes.clear();
    output_shapes.push_back(input_shapes[0]);
    
    return core::Status::SUCCESS;
}

// Register ReLU operator
REGISTER_OPERATOR(ReLU, ReLU);

} // namespace operators
} // namespace mini_infer
