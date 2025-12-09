#pragma once

#include "mini_infer/operators/operator.h"
#include "mini_infer/operators/activation_type.h"
#include "mini_infer/core/tensor.h"

namespace mini_infer {
namespace operators {

/**
 * @brief 2D Convolution Layer Operator
 * 
 * Reference: TensorRT IConvolutionLayer
 * 
 * Performs: output = Activation(Conv2D(input, weight) + bias)
 * 
 * TensorRT-style features:
 * - Built-in activation support (setActivation)
 * - Automatic kernel fusion when activation is set
 * 
 * Input shapes:
 *   - input: [N, C_in, H_in, W_in] (NCHW format)
 *   - weight: [C_out, C_in/groups, kernel_h, kernel_w]
 *   - bias (optional): [C_out]
 * 
 * Output shape:
 *   - output: [N, C_out, H_out, W_out]
 *   where:
 *     H_out = (H_in + 2*padding_h - dilation_h*(kernel_h-1) - 1) / stride_h + 1
 *     W_out = (W_in + 2*padding_w - dilation_w*(kernel_w-1) - 1) / stride_w + 1
 */

struct Conv2DParam : public OpParam {
    int kernel_h;      // Kernel height
    int kernel_w;      // Kernel width
    int stride_h;      // Stride in height dimension
    int stride_w;      // Stride in width dimension
    int padding_h;     // Padding in height dimension
    int padding_w;     // Padding in width dimension
    int dilation_h;    // Dilation in height dimension
    int dilation_w;    // Dilation in width dimension
    int groups;        // Number of groups for grouped convolution
    bool use_bias;     // Whether to use bias
    
    // TensorRT-style: Built-in activation
    ActivationParam activation;
    
    Conv2DParam()
        : kernel_h(3)
        , kernel_w(3)
        , stride_h(1)
        , stride_w(1)
        , padding_h(0)
        , padding_w(0)
        , dilation_h(1)
        , dilation_w(1)
        , groups(1)
        , use_bias(true)
        , activation(ActivationType::NONE) {}
    
    Conv2DParam(int kh, int kw, int sh = 1, int sw = 1, 
                int ph = 0, int pw = 0, int g = 1, bool bias = true)
        : kernel_h(kh)
        , kernel_w(kw)
        , stride_h(sh)
        , stride_w(sw)
        , padding_h(ph)
        , padding_w(pw)
        , dilation_h(1)
        , dilation_w(1)
        , groups(g)
        , use_bias(bias)
        , activation(ActivationType::NONE) {}
};

class Conv2D : public Operator {
public:
    explicit Conv2D(const Conv2DParam& param = Conv2DParam());
    ~Conv2D() override = default;
    
    /**
     * @brief Forward computation
     * 
     * @param inputs Input tensor list:
     *   - inputs[0]: input tensor [N, C_in, H_in, W_in]
     *   - inputs[1]: weight tensor [C_out, C_in/groups, kernel_h, kernel_w]
     *   - inputs[2]: bias tensor [C_out] (optional, if use_bias=true)
     * @param outputs Output tensor list (one output):
     *   - outputs[0]: output tensor [N, C_out, H_out, W_out]
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
     * @brief Get Conv2D parameters
     */
    const Conv2DParam& param() const { return param_; }
    
    /**
     * @brief Set Conv2D parameters
     */
    void set_param(const Conv2DParam& param) { param_ = param; }
    
    /**
     * @brief Set activation (TensorRT-style API)
     * 
     * Similar to TensorRT's IConvolutionLayer::setActivation()
     * Enables automatic kernel fusion during forward pass
     * 
     * @param type Activation type
     * @param alpha Alpha parameter (for LeakyReLU, ELU, etc.)
     * @param beta Beta parameter (for Clip, HardSigmoid, etc.)
     */
    void set_activation(ActivationType type, float alpha = 0.0f, float beta = 0.0f) {
        param_.activation = ActivationParam(type, alpha, beta);
    }
    
    /**
     * @brief Get activation parameters
     */
    const ActivationParam& get_activation() const {
        return param_.activation;
    }
    
    /**
     * @brief Check if activation is enabled
     */
    bool has_activation() const {
        return param_.activation.is_enabled();
    }

private:
    Conv2DParam param_;
};

} // namespace operators
} // namespace mini_infer

