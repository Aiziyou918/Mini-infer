#pragma once

#include "mini_infer/operators/operator.h"
#include "mini_infer/core/tensor.h"

namespace mini_infer {
namespace operators {

/**
 * @brief Pooling Type
 * 
 * Reference: TensorRT PoolingType
 */
enum class PoolingType {
    MAX,      // Max pooling
    AVERAGE   // Average pooling (excluding padding)
};

/**
 * @brief 2D Pooling Layer Operator
 * 
 * Reference: TensorRT IPoolingLayer
 * 
 * Performs spatial pooling operations:
 * - MaxPool: Takes maximum value in pooling window
 * - AvgPool: Takes average value in pooling window (excluding padding)
 * 
 * Input shape:
 *   - input: [N, C, H_in, W_in] (NCHW format)
 * 
 * Output shape:
 *   - output: [N, C, H_out, W_out]
 *   where:
 *     H_out = (H_in + 2*padding_h - kernel_h) / stride_h + 1
 *     W_out = (W_in + 2*padding_w - kernel_w) / stride_w + 1
 * 
 * Features aligned with TensorRT:
 * - Support MaxPool and AvgPool
 * - Support asymmetric kernel/stride/padding
 * - NCHW data layout
 * - Average pooling excludes padding (count_include_pad=false)
 */

struct PoolingParam : public OpParam {
    PoolingType type;   // Pooling type (MAX or AVERAGE)
    int kernel_h;       // Kernel height
    int kernel_w;       // Kernel width
    int stride_h;       // Stride in height dimension
    int stride_w;       // Stride in width dimension
    int padding_h;      // Padding in height dimension
    int padding_w;      // Padding in width dimension
    
    PoolingParam()
        : type(PoolingType::MAX)
        , kernel_h(2)
        , kernel_w(2)
        , stride_h(2)
        , stride_w(2)
        , padding_h(0)
        , padding_w(0) {}
    
    PoolingParam(PoolingType t, int kh, int kw, 
                 int sh = -1, int sw = -1,
                 int ph = 0, int pw = 0)
        : type(t)
        , kernel_h(kh)
        , kernel_w(kw)
        , stride_h(sh < 0 ? kh : sh)  // Default stride = kernel_size
        , stride_w(sw < 0 ? kw : sw)
        , padding_h(ph)
        , padding_w(pw) {}
};

class Pooling : public Operator {
public:
    explicit Pooling(const PoolingParam& param);
    
    ~Pooling() override = default;
    
    /**
     * @brief Forward computation
     * 
     * @param inputs Input tensor list:
     *   - inputs[0]: input tensor [N, C, H_in, W_in]
     * @param outputs Output tensor list:
     *   - outputs[0]: output tensor [N, C, H_out, W_out]
     * @return Execution status
     */
    core::Status forward(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs
    ) override;
    
    /**
     * @brief Infer output shape
     * 
     * @param input_shapes Input tensor shapes
     * @param output_shapes Inferred output tensor shapes
     * @return Status
     */
    core::Status infer_shape(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes
    ) override;
    
    const PoolingParam& param() const { return param_; }

private:
    PoolingParam param_;
};

} // namespace operators
} // namespace mini_infer
