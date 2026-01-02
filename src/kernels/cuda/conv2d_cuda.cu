#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/operators/conv2d.h"
#include "mini_infer/operators/activation_type.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace kernels {
namespace cuda {

/**
 * @brief Device function for applying activation
 *
 * CUDA version of activation functions for kernel fusion
 */
__device__ __forceinline__ float apply_activation_cuda(
    float value,
    int activation_type,
    float alpha,
    float beta
) {
    switch (activation_type) {
        case 0:  // NONE
            return value;
        case 1:  // RELU
            return value > 0.0f ? value : 0.0f;
        case 2:  // SIGMOID
            return 1.0f / (1.0f + expf(-value));
        case 3:  // TANH
            return tanhf(value);
        case 4:  // LEAKY_RELU
            return value > 0.0f ? value : alpha * value;
        case 5:  // ELU
            return value > 0.0f ? value : alpha * (expf(value) - 1.0f);
        case 6:  // SELU
            {
                constexpr float scale = 1.0507009873554804934193349852946f;
                constexpr float selu_alpha = 1.6732632423543772848170429916717f;
                return scale * (value > 0.0f ? value : selu_alpha * (expf(value) - 1.0f));
            }
        case 7:  // SOFTSIGN
            return value / (1.0f + fabsf(value));
        case 8:  // SOFTPLUS
            return logf(expf(value) + 1.0f);
        case 9:  // CLIP
            return fminf(fmaxf(value, alpha), beta);
        case 10: // HARD_SIGMOID
            return fmaxf(0.0f, fminf(1.0f, alpha * value + beta));
        case 11: // SCALED_TANH
            return alpha * tanhf(beta * value);
        case 12: // THRESHOLDED_RELU
            return value > alpha ? value : 0.0f;
        default:
            return value;
    }
}

/**
 * @brief Direct convolution CUDA kernel
 *
 * Computes 2D convolution with optional bias and activation fusion.
 * Each thread computes one output element.
 * Supports: stride, padding, dilation, bias, activation fusion
 *
 * Input:  [N, C_in, H_in, W_in]
 * Weight: [C_out, C_in, K_h, K_w]
 * Bias:   [C_out] (optional)
 * Output: [N, C_out, H_out, W_out]
 */
__global__ void conv2d_direct_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    bool use_bias,
    int activation_type, float act_alpha, float act_beta
) {
    // Calculate output position
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z % C_out;
    int n = blockIdx.z / C_out;

    if (n >= N || c_out >= C_out || h_out >= H_out || w_out >= W_out) {
        return;
    }

    // Calculate input starting position
    int h_in_start = h_out * stride_h - pad_h;
    int w_in_start = w_out * stride_w - pad_w;

    // Accumulate convolution result
    float sum = 0.0f;

    // Loop over input channels and kernel
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                // Apply dilation
                int h_in = h_in_start + kh * dilation_h;
                int w_in = w_in_start + kw * dilation_w;

                // Check bounds (handle padding)
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int weight_idx = ((c_out * C_in + c_in) * K_h + kh) * K_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias
    if (use_bias) {
        sum += bias[c_out];
    }

    // Apply activation
    sum = apply_activation_cuda(sum, activation_type, act_alpha, act_beta);

    // Write output
    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[output_idx] = sum;
}

/**
 * @brief Optimized convolution kernel using shared memory
 *
 * Uses shared memory to cache input tiles, reducing global memory access.
 * Supports: stride, padding, dilation, bias, activation fusion
 */
__global__ void conv2d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    bool use_bias,
    int activation_type, float act_alpha, float act_beta
) {
    constexpr int TILE_H = 16;
    constexpr int TILE_W = 16;

    extern __shared__ float shared_input[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int w_out = blockIdx.x * TILE_W + tx;
    int h_out = blockIdx.y * TILE_H + ty;
    int c_out = blockIdx.z % C_out;
    int n = blockIdx.z / C_out;

    if (n >= N || c_out >= C_out) {
        return;
    }

    float sum = 0.0f;

    // Calculate dilated kernel size
    int K_h_dilated = (K_h - 1) * dilation_h + 1;
    int K_w_dilated = (K_w - 1) * dilation_w + 1;

    // Process one input channel at a time
    for (int c_in = 0; c_in < C_in; ++c_in) {
        int h_in_base = blockIdx.y * TILE_H * stride_h - pad_h;
        int w_in_base = blockIdx.x * TILE_W * stride_w - pad_w;

        int shared_h = (TILE_H - 1) * stride_h + K_h_dilated;
        int shared_w = (TILE_W - 1) * stride_w + K_w_dilated;

        // Cooperative loading
        int num_elements = shared_h * shared_w;
        int threads_per_block = blockDim.x * blockDim.y;
        int thread_id = ty * blockDim.x + tx;

        for (int i = thread_id; i < num_elements; i += threads_per_block) {
            int sh = i / shared_w;
            int sw = i % shared_w;
            int h_in = h_in_base + sh;
            int w_in = w_in_base + sw;

            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                shared_input[i] = input[input_idx];
            } else {
                shared_input[i] = 0.0f;
            }
        }

        __syncthreads();

        // Compute convolution
        if (h_out < H_out && w_out < W_out) {
            int sh_start = ty * stride_h;
            int sw_start = tx * stride_w;

            for (int kh = 0; kh < K_h; ++kh) {
                for (int kw = 0; kw < K_w; ++kw) {
                    int sh = sh_start + kh * dilation_h;
                    int sw = sw_start + kw * dilation_w;
                    int shared_idx = sh * shared_w + sw;
                    int weight_idx = ((c_out * C_in + c_in) * K_h + kh) * K_w + kw;
                    sum += shared_input[shared_idx] * weight[weight_idx];
                }
            }
        }

        __syncthreads();
    }

    // Write output with bias and activation
    if (h_out < H_out && w_out < W_out) {
        // Add bias
        if (use_bias) {
            sum += bias[c_out];
        }

        // Apply activation
        sum = apply_activation_cuda(sum, activation_type, act_alpha, act_beta);

        int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
        output[output_idx] = sum;
    }
}

/**
 * @brief Conv2D CUDA kernel function
 *
 * Implements 2D convolution using hand-written CUDA kernels.
 * Supports:
 * - Arbitrary padding, stride, and dilation
 * - Optional bias
 * - Activation fusion (ReLU, Sigmoid, Tanh, etc.)
 * - Automatic kernel selection based on problem size
 *
 * @param ctx Kernel execution context
 */
void conv2d_cuda(KernelContext* ctx) {
    if (!ctx || !ctx->inputs || !ctx->outputs || !ctx->device_context || !ctx->op_param) {
        return;
    }

    const auto& inputs = *ctx->inputs;
    auto& outputs = *ctx->outputs;
    if (inputs.size() < 2 || outputs.empty() || !inputs[0] || !inputs[1] || !outputs[0]) {
        return;
    }

    // Get CUDA device context
    auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(
        ctx->device_context
    );
    if (!cuda_ctx) {
        return;
    }

    // Get convolution parameters
    auto* param = static_cast<const operators::Conv2DParam*>(ctx->op_param);
    if (!param) {
        return;
    }

    // Get input, weight, and output tensors
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    auto& output = outputs[0];

    // Get tensor shapes
    const auto& in_shape = input->shape();
    const auto& out_shape = output->shape();
    const auto& w_shape = weight->shape();

    // Extract dimensions
    int N = in_shape[0];       // Batch size
    int C_in = in_shape[1];    // Input channels
    int H_in = in_shape[2];    // Input height
    int W_in = in_shape[3];    // Input width

    int C_out = out_shape[1];  // Output channels
    int H_out = out_shape[2];  // Output height
    int W_out = out_shape[3];  // Output width

    int K_h = w_shape[2];      // Kernel height
    int K_w = w_shape[3];      // Kernel width

    // Convolution parameters
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int pad_h = param->padding_h;
    int pad_w = param->padding_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;

    // Bias parameters
    bool use_bias = param->use_bias && inputs.size() > 2 && inputs[2];
    const float* bias_data = use_bias ? static_cast<const float*>(inputs[2]->data()) : nullptr;

    // Activation parameters
    int activation_type = static_cast<int>(param->activation.type);
    float act_alpha = param->activation.alpha;
    float act_beta = param->activation.beta;

    // Calculate dilated kernel size
    int K_h_dilated = (K_h - 1) * dilation_h + 1;
    int K_w_dilated = (K_w - 1) * dilation_w + 1;

    // Choose kernel based on problem size
    // Use shared memory for large outputs and small effective kernels
    bool use_shared = (H_out * W_out > 256) && (K_h_dilated * K_w_dilated <= 49);

    if (use_shared) {
        // Use shared memory kernel
        dim3 threads(16, 16);  // 256 threads per block
        dim3 blocks(
            (W_out + threads.x - 1) / threads.x,
            (H_out + threads.y - 1) / threads.y,
            N * C_out
        );

        // Calculate shared memory size
        int shared_h = (16 - 1) * stride_h + K_h_dilated;
        int shared_w = (16 - 1) * stride_w + K_w_dilated;
        size_t shared_mem_size = shared_h * shared_w * sizeof(float);

        conv2d_shared_kernel<<<blocks, threads, shared_mem_size, cuda_ctx->stream()>>>(
            static_cast<const float*>(input->data()),
            static_cast<const float*>(weight->data()),
            bias_data,
            static_cast<float*>(output->data()),
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            K_h, K_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            use_bias,
            activation_type, act_alpha, act_beta
        );
    } else {
        // Use direct kernel
        dim3 threads(16, 16);  // 256 threads per block
        dim3 blocks(
            (W_out + threads.x - 1) / threads.x,
            (H_out + threads.y - 1) / threads.y,
            N * C_out
        );

        conv2d_direct_kernel<<<blocks, threads, 0, cuda_ctx->stream()>>>(
            static_cast<const float*>(input->data()),
            static_cast<const float*>(weight->data()),
            bias_data,
            static_cast<float*>(output->data()),
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            K_h, K_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            use_bias,
            activation_type, act_alpha, act_beta
        );
    }

    // Check for kernel launch errors
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDA] " + std::string(cudaGetErrorString(status)) +
                     " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
        return;
    }
}

/**
 * @brief Conv2D CUDA kernel registrar
 *
 * Automatically registers the Conv2D CUDA kernel on program startup.
 */
namespace {
    struct Conv2DCUDARegistrar {
        Conv2DCUDARegistrar() {
            KernelRegistry::instance().register_kernel(
                core::OpType::kCONVOLUTION,
                core::DeviceType::CUDA,
                core::DataType::FLOAT32,
                conv2d_cuda
            );
        }
    };
    static Conv2DCUDARegistrar g_conv2d_cuda_registrar;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace mini_infer
