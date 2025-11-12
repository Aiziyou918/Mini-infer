#pragma once

#include "mini_infer/operators/operator.h"

namespace mini_infer {
namespace operators {

/**
 * @brief Conv2D 算子参数
 */
struct Conv2DParam : public OpParam {
    int32_t kernel_h{3};
    int32_t kernel_w{3};
    int32_t stride_h{1};
    int32_t stride_w{1};
    int32_t padding_h{0};
    int32_t padding_w{0};
    int32_t dilation_h{1};
    int32_t dilation_w{1};
    int32_t groups{1};
};

/**
 * @brief 2D 卷积算子
 */
class Conv2D : public Operator {
public:
    Conv2D();
    ~Conv2D() override = default;
    
    core::Status forward(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs
    ) override;
    
    core::Status infer_shape(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes
    ) override;
};

} // namespace operators
} // namespace mini_infer

