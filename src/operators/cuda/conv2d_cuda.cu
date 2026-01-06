#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/operators/activation_type.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace operators {

namespace {

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
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z % C_out;
    int n = blockIdx.z / C_out;

    if (n >= N || c_out >= C_out || h_out >= H_out || w_out >= W_out) {
        return;
    }

    int h_in_start = h_out * stride_h - pad_h;
    int w_in_start = w_out * stride_w - pad_w;

    float sum = 0.0f;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int h_in = h_in_start + kh * dilation_h;
                int w_in = w_in_start + kw * dilation_w;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int weight_idx = ((c_out * C_in + c_in) * K_h + kh) * K_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (use_bias) {
        sum += bias[c_out];
    }

    sum = apply_activation_cuda(sum, activation_type, act_alpha, act_beta);

    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[output_idx] = sum;
}

/**
 * @brief Optimized convolution kernel using shared memory
 *
 * Uses shared memory to cache input tiles, reducing global memory access.
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

    int K_h_dilated = (K_h - 1) * dilation_h + 1;
    int K_w_dilated = (K_w - 1) * dilation_w + 1;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        int h_in_base = blockIdx.y * TILE_H * stride_h - pad_h;
        int w_in_base = blockIdx.x * TILE_W * stride_w - pad_w;

        int shared_h = (TILE_H - 1) * stride_h + K_h_dilated;
        int shared_w = (TILE_W - 1) * stride_w + K_w_dilated;

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

    if (h_out < H_out && w_out < W_out) {
        if (use_bias) {
            sum += bias[c_out];
        }

        sum = apply_activation_cuda(sum, activation_type, act_alpha, act_beta);

        int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
        output[output_idx] = sum;
    }
}

}  // namespace

/**
 * @brief Conv2D CUDA Plugin
 *
 * Performs: output = Activation(Conv2D(input, weight) + bias)
 * Uses hand-written CUDA kernels with automatic kernel selection.
 *
 * Input shapes:
 *   - input: [N, C_in, H_in, W_in] (NCHW format)
 *   - weight: [C_out, C_in/groups, kernel_h, kernel_w]
 *   - bias (optional): [C_out]
 *
 * Output shape:
 *   - output: [N, C_out, H_out, W_out]
 */
class Conv2DCUDAPlugin : public CUDAPlugin<Conv2DCUDAPlugin, Conv2DParam> {
public:
    Conv2DCUDAPlugin() {
        param_ = std::make_shared<Conv2DParam>();
    }
    ~Conv2DCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Conv";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kCONVOLUTION;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return param_ && param_->use_bias ? 3 : 2;
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {

        if (input_shapes.size() < 2) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_shape = input_shapes[0];
        const auto& weight_shape = input_shapes[1];

        if (input_shape.ndim() != 4 || weight_shape.ndim() != 4) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        int64_t N = input_shape[0];
        int64_t C_in = input_shape[1];
        int64_t H_in = input_shape[2];
        int64_t W_in = input_shape[3];

        int64_t C_out = weight_shape[0];
        int64_t C_in_per_group = weight_shape[1];
        int64_t kernel_h = weight_shape[2];
        int64_t kernel_w = weight_shape[3];

        int groups = param_ ? param_->groups : 1;
        if (groups != 1) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }
        if (C_in != C_in_per_group * groups) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        if (param_ && param_->use_bias && input_shapes.size() > 2) {
            const auto& bias_shape = input_shapes[2];
            if (bias_shape.ndim() != 1 || bias_shape[0] != C_out) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
        }

        int pad_h = param_ ? param_->padding_h : 0;
        int pad_w = param_ ? param_->padding_w : 0;
        int stride_h = param_ ? param_->stride_h : 1;
        int stride_w = param_ ? param_->stride_w : 1;
        int dilation_h = param_ ? param_->dilation_h : 1;
        int dilation_w = param_ ? param_->dilation_w : 1;

        int64_t H_out = (H_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        int64_t W_out = (W_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

        output_shapes.clear();
        output_shapes.push_back(core::Shape({N, C_out, H_out, W_out}));
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {

        if (inputs.size() < 2 || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input = inputs[0];
        const auto& weight = inputs[1];
        auto& output = outputs[0];

        if (!input || !weight || !output || !context.device_context) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(
            context.device_context
        );
        if (!cuda_ctx) {
            return core::Status::ERROR_BACKEND;
        }

        if (input->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& in_shape = input->shape();
        const auto& out_shape = output->shape();
        const auto& w_shape = weight->shape();

        int N = static_cast<int>(in_shape[0]);
        int C_in = static_cast<int>(in_shape[1]);
        int H_in = static_cast<int>(in_shape[2]);
        int W_in = static_cast<int>(in_shape[3]);

        int C_out = static_cast<int>(out_shape[1]);
        int H_out = static_cast<int>(out_shape[2]);
        int W_out = static_cast<int>(out_shape[3]);

        int K_h = static_cast<int>(w_shape[2]);
        int K_w = static_cast<int>(w_shape[3]);

        int stride_h = param_->stride_h;
        int stride_w = param_->stride_w;
        int pad_h = param_->padding_h;
        int pad_w = param_->padding_w;
        int dilation_h = param_->dilation_h;
        int dilation_w = param_->dilation_w;

        bool use_bias = param_->use_bias && inputs.size() > 2 && inputs[2];
        const float* bias_data = use_bias ? static_cast<const float*>(inputs[2]->data()) : nullptr;

        int activation_type = static_cast<int>(param_->activation);
        float act_alpha = 0.0f;  // Default alpha for activations
        float act_beta = 0.0f;   // Default beta for activations

        int K_h_dilated = (K_h - 1) * dilation_h + 1;
        int K_w_dilated = (K_w - 1) * dilation_w + 1;

        // Choose kernel based on problem size
        bool use_shared = (H_out * W_out > 256) && (K_h_dilated * K_w_dilated <= 49);

        if (use_shared) {
            dim3 threads(16, 16);
            dim3 blocks(
                (W_out + threads.x - 1) / threads.x,
                (H_out + threads.y - 1) / threads.y,
                N * C_out
            );

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
            dim3 threads(16, 16);
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

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] Conv2D plugin error: " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }
};

// Define creator and register plugin
REGISTER_PLUGIN_SIMPLE(Conv2DCUDAPlugin, "Conv", kCONVOLUTION, CUDA)

}  // namespace operators
}  // namespace mini_infer
