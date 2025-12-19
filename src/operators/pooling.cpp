#include "mini_infer/operators/pooling.h"

#include <stdexcept>

#include "mini_infer/core/op_type.h"
#include "mini_infer/kernels/kernel_registry.h"

namespace mini_infer {
namespace operators {

Pooling::Pooling(const PoolingParam& param)
    : Operator(param.type == PoolingType::MAX ? core::op_names::kMaxPool
                                              : core::op_names::kAveragePool),
      param_(param) {
    // Validate parameters
    if (param_.kernel_h <= 0 || param_.kernel_w <= 0) {
        throw std::invalid_argument("Pooling: kernel size must be positive");
    }

    if (param_.stride_h <= 0 || param_.stride_w <= 0) {
        throw std::invalid_argument("Pooling: stride must be positive");
    }

    if (param_.padding_h < 0 || param_.padding_w < 0) {
        throw std::invalid_argument("Pooling: padding must be non-negative");
    }
}

core::Status Pooling::forward(const std::vector<std::shared_ptr<core::Tensor>>& inputs,
                              std::vector<std::shared_ptr<core::Tensor>>& outputs) {
    // Validate inputs
    if (inputs.size() != 1) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    const auto& input = inputs[0];

    if (!input) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Check input shape (must be 4D: NCHW)
    const auto& input_shape = input->shape();
    if (input_shape.ndim() != 4) {
        return core::Status::ERROR_INVALID_ARGUMENT;
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

    const auto dtype = input->dtype();
    const auto op_type = (param_.type == PoolingType::MAX) ? core::OpType::kMAX_POOL
                                                           : core::OpType::kAVERAGE_POOL;

    kernels::KernelContext ctx;
    ctx.inputs = &inputs;
    ctx.outputs = &outputs;
    ctx.op_param = &param_;
    ctx.device_context = kernels::get_current_device_context();

    auto kernel =
        kernels::KernelRegistry::instance().find(op_type, input->device(), dtype);
    if (!kernel) {
        return core::Status::ERROR_NOT_IMPLEMENTED;
    }

    kernel(&ctx);

    return core::Status::SUCCESS;
}

core::Status Pooling::infer_shape(const std::vector<core::Shape>& input_shapes,
                                  std::vector<core::Shape>& output_shapes) {
    // Validate input
    if (input_shapes.size() != 1) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    const auto& input_shape = input_shapes[0];
    if (input_shape.ndim() != 4) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Extract dimensions
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];

    // Validate dimensions
    if (H_in <= 0 || W_in <= 0) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Calculate output dimensions
    int64_t H_out = (H_in + 2 * param_.padding_h - param_.kernel_h) / param_.stride_h + 1;
    int64_t W_out = (W_in + 2 * param_.padding_w - param_.kernel_w) / param_.stride_w + 1;

    if (H_out <= 0 || W_out <= 0) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    output_shapes.clear();
    output_shapes.push_back(core::Shape({N, C, H_out, W_out}));

    return core::Status::SUCCESS;
}

}  // namespace operators
}  // namespace mini_infer
