#include "mini_infer/operators/reshape.h"

#include "mini_infer/core/op_type.h"
#include "mini_infer/kernels/kernel_registry.h"

namespace mini_infer {
namespace operators {

// Register Reshape operator
REGISTER_OPERATOR(Reshape, Reshape);

Reshape::Reshape(const ReshapeParam& param)
    : Operator(core::op_names::kReshape, core::OpType::kRESHAPE), param_(param) {}

core::Status Reshape::forward(const std::vector<std::shared_ptr<core::Tensor>>& inputs,
                              std::vector<std::shared_ptr<core::Tensor>>& outputs) {
    // Input validation
    if (inputs.empty()) {
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

    kernels::KernelContext ctx;
    ctx.inputs = &inputs;
    ctx.outputs = &outputs;
    ctx.device_context = kernels::get_current_device_context();

    auto kernel = cached_kernel();
    if (!kernel) {
        kernel = kernels::KernelRegistry::instance().find(core::OpType::kRESHAPE, input->device(),
                                                          input->dtype());
        if (kernel) {
            set_cached_kernel(kernel);
        }
    }
    if (!kernel) {
        return core::Status::ERROR_NOT_IMPLEMENTED;
    }

    kernel(&ctx);

    return core::Status::SUCCESS;
}

core::Status Reshape::infer_shape(const std::vector<core::Shape>& input_shapes,
                                  std::vector<core::Shape>& output_shapes) {
    // Input validation
    if (input_shapes.empty()) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    const auto& input_shape = input_shapes[0];
    int64_t total_elements = input_shape.numel();

    // If we don't have target shape yet, we can't infer
    if (param_.shape.empty()) {
        // Try to get from second input shape if available
        if (input_shapes.size() >= 2) {
            // Shape is provided as tensor, we'll resolve at runtime
            // For now, just return error - need actual shape values
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Resolve shape
    std::vector<int64_t> resolved_shape;
    auto status = resolve_shape(param_.shape, total_elements, resolved_shape);
    if (status != core::Status::SUCCESS) {
        return status;
    }

    // Set output shape
    output_shapes.clear();
    output_shapes.push_back(core::Shape(resolved_shape));

    return core::Status::SUCCESS;
}

core::Status Reshape::resolve_shape(const std::vector<int64_t>& target_shape,
                                    int64_t total_elements,
                                    std::vector<int64_t>& resolved_shape) const {
    resolved_shape = target_shape;

    // Find -1 dimension (if any)
    int negative_idx = -1;
    int64_t product = 1;

    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == -1) {
            if (negative_idx != -1) {
                // Multiple -1 dimensions not allowed
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
            negative_idx = static_cast<int>(i);
        } else if (target_shape[i] == 0) {
            // Handle allowzero
            if (!param_.allowzero) {
                // 0 means copy from input shape (not implemented for simplicity)
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
            // If allowzero=true, 0 is kept as 0
            product *= 0;
        } else if (target_shape[i] < 0) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        } else {
            product *= target_shape[i];
        }
    }

    // If there's a -1 dimension, infer it
    if (negative_idx != -1) {
        if (product == 0) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        if (total_elements % product != 0) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        resolved_shape[negative_idx] = total_elements / product;
    }

    return core::Status::SUCCESS;
}

}  // namespace operators
}  // namespace mini_infer
