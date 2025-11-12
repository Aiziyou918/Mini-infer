#include "mini_infer/operators/conv2d.h"
#include <cstring>

namespace mini_infer {
namespace operators {

Conv2D::Conv2D() : Operator("Conv2D") {}

core::Status Conv2D::forward(
    const std::vector<std::shared_ptr<core::Tensor>>& inputs,
    std::vector<std::shared_ptr<core::Tensor>>& outputs) {
    
    // 输入验证
    if (inputs.size() < 2) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    
    if (!input || !weight) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // TODO: 实现实际的卷积计算
    // 这里只是框架代码，具体的卷积实现需要后续完成
    
    return core::Status::SUCCESS;
}

core::Status Conv2D::infer_shape(
    const std::vector<core::Shape>& input_shapes,
    std::vector<core::Shape>& output_shapes) {
    
    if (input_shapes.size() < 2) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    const auto& input_shape = input_shapes[0];
    const auto& weight_shape = input_shapes[1];
    
    if (input_shape.ndim() != 4 || weight_shape.ndim() != 4) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    auto* param = dynamic_cast<Conv2DParam*>(param_.get());
    if (!param) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // NCHW 格式
    int64_t N = input_shape[0];
    int64_t C_in = input_shape[1];
    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];
    
    int64_t C_out = weight_shape[0];
    
    // 计算输出尺寸
    int64_t H_out = (H_in + 2 * param->padding_h - param->dilation_h * (param->kernel_h - 1) - 1) 
                    / param->stride_h + 1;
    int64_t W_out = (W_in + 2 * param->padding_w - param->dilation_w * (param->kernel_w - 1) - 1) 
                    / param->stride_w + 1;
    
    output_shapes.clear();
    output_shapes.push_back(core::Shape({N, C_out, H_out, W_out}));
    
    return core::Status::SUCCESS;
}

} // namespace operators
} // namespace mini_infer

