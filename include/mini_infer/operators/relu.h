#pragma once

#include "mini_infer/operators/operator.h"

namespace mini_infer {
namespace operators {

/**
 * @brief ReLU Activation Operator
 * 
 * ReLU(x) = max(0, x)
 * Apply ReLU activation function to each element of the input tensor
 */
class ReLU : public Operator {
public:
    ReLU();
    ~ReLU() override = default;
    
    /**
     * @brief Forward computation
     * @param inputs Input tensor list (only one input)
     * @param outputs Output tensor list (one output)
     * @return Execution status
     */
    core::Status forward(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs
    ) override;
    
    /**
     * @brief Infer output shape
     * @param input_shapes Input shape list (only one input)
     * @param output_shapes Output shape list (one output)
     * @return Execution status
     */
    core::Status infer_shape(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes
    ) override;
};

} // namespace operators
} // namespace mini_infer
