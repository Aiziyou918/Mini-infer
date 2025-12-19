#include "mini_infer/operators/conv2d.h"

#include <vector>

#include "mini_infer/core/op_type.h"
#include "mini_infer/kernels/kernel_registry.h"

namespace mini_infer {
namespace operators {

Conv2D::Conv2D(const Conv2DParam& param) : Operator(core::op_names::kConv), param_(param) {}

core::Status Conv2D::forward(const std::vector<std::shared_ptr<core::Tensor>>& inputs,
                             std::vector<std::shared_ptr<core::Tensor>>& outputs) {
    // Input validation
    size_t expected_inputs = param_.use_bias ? 3 : 2;
    if (inputs.size() != expected_inputs) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    const auto& input = inputs[0];   // [N, C_in, H_in, W_in]
    const auto& weight = inputs[1];  // [C_out, C_in/groups, kernel_h, kernel_w]

    if (!input || !weight) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Validate bias if needed
    if (param_.use_bias) {
        const auto& bias = inputs[2];
        if (!bias) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }

    // Get input shapes
    const auto& input_shape = input->shape();
    const auto& weight_shape = weight->shape();

    // Validate shapes
    if (input_shape.ndim() != 4 || weight_shape.ndim() != 4) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Validate parameters
    if (param_.groups != 1) {
        return core::Status::ERROR_NOT_IMPLEMENTED;
    }

    // Get pre-allocated output tensor (Engine already did shape inference)
    if (outputs.empty() || !outputs[0]) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    auto output = outputs[0];

    // Extract output dimensions from pre-allocated tensor
    const auto& output_shape = output->shape();
    if (output_shape.ndim() != 4) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    kernels::KernelContext ctx;
    ctx.inputs = &inputs;
    ctx.outputs = &outputs;
    ctx.op_param = &param_;
    ctx.device_context = kernels::get_current_device_context();

    auto kernel = kernels::KernelRegistry::instance().find(core::OpType::kCONVOLUTION,
                                                           input->device(), input->dtype());
    if (!kernel) {
        return core::Status::ERROR_NOT_IMPLEMENTED;
    }

    kernel(&ctx);

    return core::Status::SUCCESS;
}

core::Status Conv2D::infer_shape(const std::vector<core::Shape>& input_shapes,
                                 std::vector<core::Shape>& output_shapes) {
    // Validate input count
    size_t expected_inputs = param_.use_bias ? 3 : 2;
    if (input_shapes.size() != expected_inputs) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    const auto& input_shape = input_shapes[0];   // [N, C_in, H_in, W_in]
    const auto& weight_shape = input_shapes[1];  // [C_out, C_in/groups, kernel_h, kernel_w]

    // Validate shapes
    if (input_shape.ndim() != 4 || weight_shape.ndim() != 4) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Extract dimensions
    int64_t N = input_shape[0];
    int64_t C_in = input_shape[1];
    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];

    int64_t C_out = weight_shape[0];
    int64_t C_in_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Validate channel dimensions
    // TODO: Groups convolution not yet implemented, only groups=1 supported
    if (param_.groups != 1) {
        return core::Status::ERROR_NOT_IMPLEMENTED;
    }
    if (C_in != C_in_per_group * param_.groups) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Validate bias shape if needed
    if (param_.use_bias && input_shapes.size() > 2) {
        const auto& bias_shape = input_shapes[2];
        if (bias_shape.ndim() != 1 || bias_shape[0] != C_out) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }

    // Calculate output dimensions
    int64_t H_out =
        (H_in + 2 * param_.padding_h - param_.dilation_h * (kernel_h - 1) - 1) / param_.stride_h +
        1;
    int64_t W_out =
        (W_in + 2 * param_.padding_w - param_.dilation_w * (kernel_w - 1) - 1) / param_.stride_w +
        1;

    output_shapes.clear();
    output_shapes.push_back(core::Shape({N, C_out, H_out, W_out}));

    return core::Status::SUCCESS;
}

// Register Conv2D operator
REGISTER_OPERATOR(Conv2D, Conv2D);

}  // namespace operators
}  // namespace mini_infer
