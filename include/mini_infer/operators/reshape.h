#pragma once

#include "mini_infer/operators/operator.h"
#include "mini_infer/core/tensor.h"
#include <vector>

namespace mini_infer {
namespace operators {

/**
 * @brief Reshape operator parameter
 */
struct ReshapeParam {
    std::vector<int64_t> shape;  // Target shape (-1 means infer from data)
    bool allowzero = false;      // If true, 0 in shape means keep that dimension as 0
    
    ReshapeParam() = default;
    explicit ReshapeParam(const std::vector<int64_t>& s) : shape(s) {}
};

/**
 * @brief Reshape operator
 * 
 * Reshapes input tensor to target shape without copying data.
 * Supports -1 in shape to infer dimension size.
 * 
 * ONNX Reshape specification:
 * - Input 0: data (tensor)
 * - Input 1: shape (1D tensor of int64, can be constant)
 * - Output: reshaped tensor (shares data with input)
 * 
 * Example:
 *   Input: [2, 3, 4] shape
 *   Target shape: [2, -1]
 *   Output: [2, 12] shape (inferred -1 = 3*4 = 12)
 */
class Reshape : public Operator {
public:
    explicit Reshape(const ReshapeParam& param = ReshapeParam());
    ~Reshape() override = default;

    /**
     * @brief Forward pass
     * 
     * @param inputs Input tensors:
     *   - inputs[0]: data tensor to reshape
     *   - inputs[1]: (optional) shape tensor (int64), if not provided uses param.shape
     * @param outputs Output tensors (reshaped view)
     * @return Status code
     */
    core::Status forward(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs) override;

    /**
     * @brief Infer output shape
     * 
     * @param input_shapes Input shapes
     * @param output_shapes Output shapes (inferred)
     * @return Status code
     */
    core::Status infer_shape(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) override;

    /**
     * @brief Get parameters
     */
    const ReshapeParam& param() const { return param_; }
    
    /**
     * @brief Set parameters
     */
    void set_param(const ReshapeParam& param) { param_ = param; }

private:
    ReshapeParam param_;

    /**
     * @brief Resolve shape with -1 (infer dimension)
     * 
     * @param target_shape Target shape (may contain -1)
     * @param total_elements Total number of elements in input
     * @param resolved_shape Output resolved shape
     * @return Status code
     */
    core::Status resolve_shape(
        const std::vector<int64_t>& target_shape,
        int64_t total_elements,
        std::vector<int64_t>& resolved_shape) const;
};

} // namespace operators
} // namespace mini_infer
