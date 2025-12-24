#include "mini_infer/operators/softmax.h"

#include "mini_infer/core/op_type.h"
#include "mini_infer/kernels/kernel_registry.h"

namespace mini_infer {
namespace operators {

Softmax::Softmax(const SoftmaxParam& param)
    : Operator(core::op_names::kSoftmax, core::OpType::kSOFTMAX), param_(param) {}

core::Status Softmax::forward(const std::vector<std::shared_ptr<core::Tensor>>& inputs,
                              std::vector<std::shared_ptr<core::Tensor>>& outputs) {
    if (inputs.size() != 1 || !inputs[0]) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    if (outputs.empty() || !outputs[0]) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    kernels::KernelContext ctx;
    ctx.inputs = &inputs;
    ctx.outputs = &outputs;
    ctx.op_param = &param_;
    ctx.device_context = kernels::get_current_device_context();

    auto kernel = cached_kernel();
    if (!kernel) {
        kernel = kernels::KernelRegistry::instance().find(core::OpType::kSOFTMAX,
                                                          inputs[0]->device(),
                                                          inputs[0]->dtype());
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

core::Status Softmax::infer_shape(const std::vector<core::Shape>& input_shapes,
                                  std::vector<core::Shape>& output_shapes) {
    if (input_shapes.size() != 1) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    output_shapes.clear();
    output_shapes.push_back(input_shapes[0]);
    return core::Status::SUCCESS;
}

REGISTER_OPERATOR(Softmax, Softmax);

}  // namespace operators
}  // namespace mini_infer
