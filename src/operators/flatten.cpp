#include "mini_infer/operators/flatten.h"
#include <cstring>

namespace mini_infer {
namespace operators {

Flatten::Flatten(const FlattenParam& param)
    : Operator("Flatten"), param_(param) {}

core::Status Flatten::forward(
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
    
    // Calculate output shape
    std::vector<core::Shape> input_shapes = {input_shape};
    std::vector<core::Shape> output_shapes;
    auto status = infer_shape(input_shapes, output_shapes);
    if (status != core::Status::SUCCESS) {
        return status;
    }
    
    // Flatten is just a view change - create a view with different shape
    // This is a ZERO-COPY operation! (shares the same underlying data)
    auto output = input->view(output_shapes[0]);
    if (!output) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Set output
    outputs.clear();
    outputs.push_back(output);
    
    return core::Status::SUCCESS;
}

core::Status Flatten::infer_shape(
    const std::vector<core::Shape>& input_shapes,
    std::vector<core::Shape>& output_shapes) {
    
    // Input validation
    if (input_shapes.empty()) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    const auto& input_shape = input_shapes[0];
    const auto& dims = input_shape.dims();
    
    if (dims.empty()) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Normalize axis
    int axis = param_.axis;
    if (axis < 0) {
        axis += static_cast<int>(dims.size());
    }
    
    if (axis < 0 || axis >= static_cast<int>(dims.size())) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Calculate output shape: [dim_0 * ... * dim_{axis-1}, dim_axis * ... * dim_n]
    int64_t first_dim = 1;
    for (int i = 0; i < axis; ++i) {
        first_dim *= dims[i];
    }
    
    int64_t second_dim = 1;
    for (size_t i = axis; i < dims.size(); ++i) {
        second_dim *= dims[i];
    }
    
    // Create output shape
    output_shapes.clear();
    output_shapes.push_back(core::Shape({first_dim, second_dim}));
    
    return core::Status::SUCCESS;
}

// Register Flatten operator
REGISTER_OPERATOR(Flatten, Flatten);

} // namespace operators
} // namespace mini_infer
