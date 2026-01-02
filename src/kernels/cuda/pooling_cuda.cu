#include "mini_infer/kernels/pooling.h"
#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/operators/pooling.h"
#include "mini_infer/utils/logger.h"

#include <string>
#include <cfloat>

namespace mini_infer {
namespace kernels {
namespace cuda {

/**
 * @brief MaxPool2D CUDA kernel
 *
 * Each thread computes one output element.
 * Handles padding by treating padded areas as -infinity.
 */
__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    int total = N * C * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        // Decode output index
        int w_out = i % W_out;
        int h_out = (i / W_out) % H_out;
        int c = (i / (W_out * H_out)) % C;
        int n = i / (W_out * H_out * C);

        // Calculate input window boundaries
        int h_start = h_out * stride_h - padding_h;
        int w_start = w_out * stride_w - padding_w;
        int h_end = min(h_start + kernel_h, H_in);
        int w_end = min(w_start + kernel_w, W_in);
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);

        // Find maximum in window
        float max_val = -FLT_MAX;
        const float* input_nc = input + (n * C + c) * H_in * W_in;

        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                float val = input_nc[h * W_in + w];
                max_val = fmaxf(max_val, val);
            }
        }

        output[i] = max_val;
    }
}

/**
 * @brief AvgPool2D CUDA kernel
 *
 * Average pooling excludes padding (count_include_pad=false).
 */
__global__ void avgpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    int total = N * C * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        // Decode output index
        int w_out = i % W_out;
        int h_out = (i / W_out) % H_out;
        int c = (i / (W_out * H_out)) % C;
        int n = i / (W_out * H_out * C);

        // Calculate input window boundaries
        int h_start = h_out * stride_h - padding_h;
        int w_start = w_out * stride_w - padding_w;
        int h_end = min(h_start + kernel_h, H_in);
        int w_end = min(w_start + kernel_w, W_in);
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);

        // Calculate sum and count (excluding padding)
        float sum = 0.0f;
        int count = 0;
        const float* input_nc = input + (n * C + c) * H_in * W_in;

        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                sum += input_nc[h * W_in + w];
                ++count;
            }
        }

        output[i] = (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
    }
}

/**
 * @brief MaxPool2D CUDA implementation
 */
void maxpool2d_cuda_impl(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    int total = N * C * H_out * W_out;
    const int threads = 256;
    int blocks = std::min((total + threads - 1) / threads, 65535);

    cudaStream_t stream = nullptr;
    auto* ctx = get_current_device_context();
    if (ctx) {
        auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(ctx);
        stream = cuda_ctx->stream();
    }

    maxpool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w
    );

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDA] MaxPool2D kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief AvgPool2D CUDA implementation
 */
void avgpool2d_cuda_impl(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    int total = N * C * H_out * W_out;
    const int threads = 256;
    int blocks = std::min((total + threads - 1) / threads, 65535);

    cudaStream_t stream = nullptr;
    auto* ctx = get_current_device_context();
    if (ctx) {
        auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(ctx);
        stream = cuda_ctx->stream();
    }

    avgpool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w
    );

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDA] AvgPool2D kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief Check if CUDA is available
 */
bool is_cuda_available_pooling() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

/**
 * @brief Pooling dispatch functions for KernelRegistry
 */
namespace {

template <typename T>
void maxpool_dispatch_cuda(KernelContext* ctx) {
    if (!ctx || !ctx->inputs || !ctx->outputs || !ctx->device_context) {
        return;
    }
    const auto& inputs = *ctx->inputs;
    auto& outputs = *ctx->outputs;
    if (inputs.empty() || outputs.empty() || !inputs[0] || !outputs[0]) {
        return;
    }
    const auto* param = ctx->param<operators::PoolingParam>();
    if (!param) {
        return;
    }

    const auto& input_shape = inputs[0]->shape();
    const auto& output_shape = outputs[0]->shape();
    if (input_shape.ndim() != 4 || output_shape.ndim() != 4) {
        return;
    }

    const T* input_data = static_cast<const T*>(inputs[0]->data());
    T* output_data = static_cast<T*>(outputs[0]->data());

    PoolingKernel::maxpool2d<T>(
        input_data, output_data,
        static_cast<int>(input_shape[0]),
        static_cast<int>(input_shape[1]),
        static_cast<int>(input_shape[2]),
        static_cast<int>(input_shape[3]),
        static_cast<int>(output_shape[2]),
        static_cast<int>(output_shape[3]),
        param->kernel_h, param->kernel_w,
        param->stride_h, param->stride_w,
        param->padding_h, param->padding_w,
        KernelBackend::CUDA
    );
}

template <typename T>
void avgpool_dispatch_cuda(KernelContext* ctx) {
    if (!ctx || !ctx->inputs || !ctx->outputs || !ctx->device_context) {
        return;
    }
    const auto& inputs = *ctx->inputs;
    auto& outputs = *ctx->outputs;
    if (inputs.empty() || outputs.empty() || !inputs[0] || !outputs[0]) {
        return;
    }
    const auto* param = ctx->param<operators::PoolingParam>();
    if (!param) {
        return;
    }

    const auto& input_shape = inputs[0]->shape();
    const auto& output_shape = outputs[0]->shape();
    if (input_shape.ndim() != 4 || output_shape.ndim() != 4) {
        return;
    }

    const T* input_data = static_cast<const T*>(inputs[0]->data());
    T* output_data = static_cast<T*>(outputs[0]->data());

    PoolingKernel::avgpool2d<T>(
        input_data, output_data,
        static_cast<int>(input_shape[0]),
        static_cast<int>(input_shape[1]),
        static_cast<int>(input_shape[2]),
        static_cast<int>(input_shape[3]),
        static_cast<int>(output_shape[2]),
        static_cast<int>(output_shape[3]),
        param->kernel_h, param->kernel_w,
        param->stride_h, param->stride_w,
        param->padding_h, param->padding_w,
        KernelBackend::CUDA
    );
}

}  // namespace

/**
 * @brief Pooling CUDA kernel registrar
 */
namespace {
    void register_pooling_cuda_kernels() {
        // Register to template-based registry
        MaxPool2DRegistry<float>::instance().register_kernel(
            KernelBackend::CUDA,
            maxpool2d_cuda_impl,
            is_cuda_available_pooling,
            200
        );

        AvgPool2DRegistry<float>::instance().register_kernel(
            KernelBackend::CUDA,
            avgpool2d_cuda_impl,
            is_cuda_available_pooling,
            200
        );
    }

    struct PoolingCUDARegistrar {
        PoolingCUDARegistrar() {
            register_pooling_cuda_kernels();

            // Register to main KernelRegistry
            KernelRegistry::instance().register_kernel(
                core::OpType::kMAX_POOL,
                core::DeviceType::CUDA,
                core::DataType::FLOAT32,
                maxpool_dispatch_cuda<float>
            );
            KernelRegistry::instance().register_kernel(
                core::OpType::kAVERAGE_POOL,
                core::DeviceType::CUDA,
                core::DataType::FLOAT32,
                avgpool_dispatch_cuda<float>
            );
        }
    };
    static PoolingCUDARegistrar g_pooling_cuda_registrar;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace mini_infer
