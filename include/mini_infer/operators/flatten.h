#pragma once

#include "mini_infer/operators/operator.h"
#include "mini_infer/core/tensor.h"

namespace mini_infer {
namespace operators {

/**
 * @brief Flatten Operator
 * 
 * Flattens the input tensor into a 2D matrix.
 * 
 * For axis=1 (default): 
 *   Input: [N, C, H, W] -> Output: [N, C*H*W]
 * For axis=k:
 *   Flattens dimensions from axis k onwards
 * 
 * Reference: ONNX Flatten operator
 */

struct FlattenParam : public OpParam {
    int axis;  // Axis to flatten from (default: 1)
    
    FlattenParam() : axis(1) {}
    explicit FlattenParam(int ax) : axis(ax) {}
};

class Flatten : public Operator {
public:
    explicit Flatten(const FlattenParam& param = FlattenParam());
    ~Flatten() override = default;
    
    /**
     * @brief Forward computation
     * 
     * @param inputs Input tensor list:
     *   - inputs[0]: input tensor (any shape)
     * @param outputs Output tensor list:
     *   - outputs[0]: flattened tensor (2D)
     * @return Execution status
     */
    core::Status forward(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs
    ) override;
    
    /**
     * @brief Infer output shape
     * 
     * @param input_shapes Input shape list
     * @param output_shapes Output shape list
     * @return Execution status
     */
    core::Status infer_shape(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes
    ) override;
    
    const FlattenParam& param() const { return param_; }
    void set_param(const FlattenParam& param) { param_ = param; }

private:
    FlattenParam param_;
};

} // namespace operators
} // namespace mini_infer
