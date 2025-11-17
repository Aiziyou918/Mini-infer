#include "mini_infer/operators/conv2d.h"
#include <cstring>
#include <algorithm>
#include <vector>

namespace mini_infer {
namespace operators {

Conv2D::Conv2D(const Conv2DParam& param) 
    : Operator("Conv2D")
    , param_(param) {
}

core::Status Conv2D::forward(
    const std::vector<std::shared_ptr<core::Tensor>>& inputs,
    std::vector<std::shared_ptr<core::Tensor>>& outputs) {
    
    // Input validation
    size_t expected_inputs = param_.use_bias ? 3 : 2;
    if (inputs.size() != expected_inputs) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    const auto& input = inputs[0];   // [N, C_in, H_in, W_in]
    const auto& weight = inputs[1];  // [C_out, C_in/groups, kernel_h, kernel_w]
    
    if (!input || !weight) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Validate bias if needed
    if (param_.use_bias) {
        const auto& bias = inputs[2];
        if (!bias) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }
    
    // Get input shapes
    const auto& input_shape = input->shape();
    const auto& weight_shape = weight->shape();
    
    // Validate shapes
    if (input_shape.ndim() != 4 || weight_shape.ndim() != 4) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Extract dimensions
    int N = static_cast<int>(input_shape[0]);
    int C_in = static_cast<int>(input_shape[1]);
    int H_in = static_cast<int>(input_shape[2]);
    int W_in = static_cast<int>(input_shape[3]);
    
    int C_out = static_cast<int>(weight_shape[0]);
    int C_in_per_group = static_cast<int>(weight_shape[1]);
    int kernel_h = static_cast<int>(weight_shape[2]);
    int kernel_w = static_cast<int>(weight_shape[3]);
    
    // Validate parameters
    if (C_in != C_in_per_group * param_.groups) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Calculate output dimensions
    int H_out = (H_in + 2 * param_.padding_h - param_.dilation_h * (kernel_h - 1) - 1) 
                / param_.stride_h + 1;
    int W_out = (W_in + 2 * param_.padding_w - param_.dilation_w * (kernel_w - 1) - 1) 
                / param_.stride_w + 1;
    
    // Create output tensor
    std::vector<int64_t> output_dims = {
        static_cast<int64_t>(N),
        static_cast<int64_t>(C_out),
        static_cast<int64_t>(H_out),
        static_cast<int64_t>(W_out)
    };
    core::Shape output_shape(output_dims);
    
    auto output = core::Tensor::create(output_shape, input->dtype());
    if (!output) {
        return core::Status::ERROR_OUT_OF_MEMORY;
    }
    
    // Perform computation based on data type
    const auto dtype = input->dtype();
    
    if (dtype == core::DataType::FLOAT32) {
        const float* input_data = static_cast<const float*>(input->data());
        const float* weight_data = static_cast<const float*>(weight->data());
        float* output_data = static_cast<float*>(output->data());
        
        // Allocate col_buffer for im2col
        int col_buffer_size = C_in * kernel_h * kernel_w * H_out * W_out;
        std::vector<float> col_buffer(col_buffer_size);
        
        // Process each batch
        for (int n = 0; n < N; ++n) {
            const float* input_n = input_data + n * C_in * H_in * W_in;
            float* output_n = output_data + n * C_out * H_out * W_out;
            
            // im2col transformation
            im2col<float>(
                input_n,
                col_buffer.data(),
                C_in, H_in, W_in,
                kernel_h, kernel_w,
                param_.stride_h, param_.stride_w,
                param_.padding_h, param_.padding_w,
                param_.dilation_h, param_.dilation_w,
                H_out, W_out
            );
            
            // GEMM: output = weight @ col_buffer
            // weight: [C_out, C_in*kh*kw]
            // col_buffer: [C_in*kh*kw, H_out*W_out]
            // output: [C_out, H_out*W_out]
            int M = C_out;
            int N_gemm = H_out * W_out;
            int K = C_in * kernel_h * kernel_w;
            
            gemm_nn<float>(
                weight_data,
                col_buffer.data(),
                output_n,
                M, N_gemm, K
            );
        }
        
        // Add bias if needed
        if (param_.use_bias) {
            const float* bias_data = static_cast<const float*>(inputs[2]->data());
            for (int n = 0; n < N; ++n) {
                float* output_n = output_data + n * C_out * H_out * W_out;
                add_bias<float>(output_n, bias_data, C_out, H_out, W_out);
            }
        }
        
    } else {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Set output
    outputs.clear();
    outputs.push_back(output);
    
    return core::Status::SUCCESS;
}

core::Status Conv2D::infer_shape(
    const std::vector<core::Shape>& input_shapes,
    std::vector<core::Shape>& output_shapes) {
    
    // Validate input count
    size_t expected_inputs = param_.use_bias ? 3 : 2;
    if (input_shapes.size() != expected_inputs) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    const auto& input_shape = input_shapes[0];   // [N, C_in, H_in, W_in]
    const auto& weight_shape = input_shapes[1];  // [C_out, C_in/groups, kernel_h, kernel_w]
    
    // Validate shapes
    if (input_shape.ndim() != 4 || weight_shape.ndim() != 4) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Extract dimensions
    int64_t N = input_shape[0];
    int64_t C_in = input_shape[1];
    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];
    
    int64_t C_out = weight_shape[0];
    int64_t C_in_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];
    
    // Validate channel dimensions
    if (C_in != C_in_per_group * param_.groups) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Validate bias shape if needed
    if (param_.use_bias && input_shapes.size() > 2) {
        const auto& bias_shape = input_shapes[2];
        if (bias_shape.ndim() != 1 || bias_shape[0] != C_out) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }
    
    // Calculate output dimensions
    int64_t H_out = (H_in + 2 * param_.padding_h - param_.dilation_h * (kernel_h - 1) - 1) 
                    / param_.stride_h + 1;
    int64_t W_out = (W_in + 2 * param_.padding_w - param_.dilation_w * (kernel_w - 1) - 1) 
                    / param_.stride_w + 1;
    
    output_shapes.clear();
    output_shapes.push_back(core::Shape({N, C_out, H_out, W_out}));
    
    return core::Status::SUCCESS;
}

// im2col implementation for convolution
// Reference: Caffe's im2col
template<typename T>
void Conv2D::im2col(
    const T* input,
    T* col_buffer,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int out_height,
    int out_width) {
    
    int channel_size = height * width;
    
    for (int c = 0; c < channels; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Calculate input row and column
                int input_row_start = -padding_h + kh * dilation_h;
                int input_col_start = -padding_w + kw * dilation_w;
                
                // Column index
                int col_idx = (c * kernel_h * kernel_w + kh * kernel_w + kw);
                
                for (int oh = 0; oh < out_height; ++oh) {
                    for (int ow = 0; ow < out_width; ++ow) {
                        // Calculate corresponding input position
                        int input_row = input_row_start + oh * stride_h;
                        int input_col = input_col_start + ow * stride_w;
                        
                        int col_buffer_idx = col_idx * out_height * out_width + oh * out_width + ow;
                        
                        // Check if position is within valid input bounds
                        if (input_row >= 0 && input_row < height &&
                            input_col >= 0 && input_col < width) {
                            int input_idx = c * channel_size + input_row * width + input_col;
                            col_buffer[col_buffer_idx] = input[input_idx];
                        } else {
                            // Padding area
                            col_buffer[col_buffer_idx] = T(0);
                        }
                    }
                }
            }
        }
    }
}

// GEMM implementation: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
template<typename T>
void Conv2D::gemm_nn(
    const T* A,
    const T* B,
    T* C,
    int M,
    int N,
    int K) {
    
    // Initialize C to zero
    std::memset(C, 0, sizeof(T) * M * N);
    
    // Naive GEMM implementation
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            T a_val = A[m * K + k];
            const T* b_row = B + k * N;
            T* c_row = C + m * N;
            
            // Vectorizable loop
            for (int n = 0; n < N; ++n) {
                c_row[n] += a_val * b_row[n];
            }
        }
    }
}

// Add bias to output
template<typename T>
void Conv2D::add_bias(
    T* output,
    const T* bias,
    int channels,
    int height,
    int width) {
    
    int spatial_size = height * width;
    
    for (int c = 0; c < channels; ++c) {
        T bias_val = bias[c];
        T* output_channel = output + c * spatial_size;
        
        for (int i = 0; i < spatial_size; ++i) {
            output_channel[i] += bias_val;
        }
    }
}

// Explicit template instantiation
template void Conv2D::im2col<float>(const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int);
template void Conv2D::gemm_nn<float>(const float*, const float*, float*, int, int, int);
template void Conv2D::add_bias<float>(float*, const float*, int, int, int);

// Register Conv2D operator
REGISTER_OPERATOR(Conv2D, Conv2D);

} // namespace operators
} // namespace mini_infer
