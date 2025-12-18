#include "mini_infer/operators/flatten.h"

#include <cstring>

#include "mini_infer/core/op_type.h"

namespace mini_infer {
namespace operators {

Flatten::Flatten(const FlattenParam& param) : Operator(core::op_names::kFlatten), param_(param) {}

core::Status Flatten::forward(const std::vector<std::shared_ptr<core::Tensor>>& inputs,
                              std::vector<std::shared_ptr<core::Tensor>>& outputs) {
    // Input validation
    if (inputs.size() != 1) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    const auto& input = inputs[0];
    if (!input) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Get pre-allocated output tensor (Engine already did shape inference)
    if (outputs.empty() || !outputs[0]) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    auto output = outputs[0];

    // Flatten is a view operation - just copy data if not sharing memory
    // In static graph mode, Engine may allocate separate buffers
    if (input->data() != output->data()) {
        // Need to copy data
        std::memcpy(output->data(), input->data(), input->size_in_bytes());
    }

    return core::Status::SUCCESS;
}

core::Status Flatten::infer_shape(const std::vector<core::Shape>& input_shapes,
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

}  // namespace operators
}  // namespace mini_infer
