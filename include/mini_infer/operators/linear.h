#pragma once

#include "mini_infer/operators/operator.h"
#include "mini_infer/core/tensor.h"

namespace mini_infer {
namespace operators {

/**
 * @brief Linear (Fully Connected) Layer Operator
 * 
 * Performs: output = input @ weight^T + bias (if bias exists)
 * Reference: TensorRT IFullyConnectedLayer
 * 
 * Input shapes:
 *   - input: [batch_size, in_features] or [..., in_features]
 *   - weight: [out_features, in_features]
 *   - bias (optional): [out_features]
 * 
 * Output shape:
 *   - output: [batch_size, out_features] or [..., out_features]
 */

struct LinearParam : public OpParam {
    int in_features;      // Number of input features
    int out_features;     // Number of output features
    bool use_bias;        // Whether to use bias
    
    LinearParam() 
        : in_features(0)
        , out_features(0)
        , use_bias(true) {}
    
    LinearParam(int in_feat, int out_feat, bool bias = true)
        : in_features(in_feat)
        , out_features(out_feat)
        , use_bias(bias) {}
};

class Linear : public Operator {
public:
    explicit Linear(const LinearParam& param = LinearParam());
    ~Linear() override = default;
    
    /**
     * @brief Forward computation
     * 
     * @param inputs Input tensor list:
     *   - inputs[0]: input tensor [batch_size, in_features]
     *   - inputs[1]: weight tensor [out_features, in_features]
     *   - inputs[2]: bias tensor [out_features] (optional, if use_bias=true)
     * @param outputs Output tensor list (one output):
     *   - outputs[0]: output tensor [batch_size, out_features]
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
     * @param output_shapes Output shape list (one output)
     * @return Execution status
     */
    core::Status infer_shape(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes
    ) override;
    
    /**
     * @brief Get Linear layer parameters
     */
    const LinearParam& param() const { return param_; }
    
    /**
     * @brief Set Linear layer parameters
     */
    void set_param(const LinearParam& param) { param_ = param; }

private:
    LinearParam param_;
};

} // namespace operators
} // namespace mini_infer
