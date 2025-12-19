#include "mini_infer/operators/relu.h"

#include <algorithm>
#include <cstring>

#include "mini_infer/core/op_type.h"
#include "mini_infer/kernels/kernel_registry.h"

namespace mini_infer {
namespace operators {

ReLU::ReLU() : Operator(core::op_names::kRelu) {}

core::Status ReLU::forward(const std::vector<std::shared_ptr<core::Tensor>>& inputs,
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

    // Get pre-allocated output tensor (Engine already did shape inference)
    if (outputs.empty() || !outputs[0]) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    auto output = outputs[0];

    // Validate output shape matches input
    const auto& output_shape = output->shape();
    if (output_shape != input_shape) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    kernels::KernelContext ctx;
    ctx.inputs = &inputs;
    ctx.outputs = &outputs;
    ctx.device_context = kernels::get_current_device_context();

    auto kernel = kernels::KernelRegistry::instance().find(core::OpType::kRELU, input->device(),
                                                           input_dtype);
    if (!kernel) {
        return core::Status::ERROR_NOT_IMPLEMENTED;
    }

    kernel(&ctx);

    return core::Status::SUCCESS;
}

core::Status ReLU::infer_shape(const std::vector<core::Shape>& input_shapes,
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

}  // namespace operators
}  // namespace mini_infer
