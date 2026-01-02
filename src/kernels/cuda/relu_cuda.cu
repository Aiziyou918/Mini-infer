#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace kernels {
namespace cuda {

/**
 * @brief ReLU forward CUDA kernel (vectorized version)
 *
 * Computes: output[i] = max(0, input[i])
 *
 * Performance optimizations:
 * - Vectorized memory access using float4 (128-bit loads/stores)
 * - Grid-stride loop for handling arbitrary sizes
 * - Coalesced memory access pattern
 *
 * @param input Input tensor data
 * @param output Output tensor data
 * @param n Number of elements
 */
__global__ void relu_forward_kernel_vectorized(
    const float* input,
    float* output,
    int64_t n
) {
    // Calculate global thread ID
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    // Process 4 elements at a time using float4 (vectorized access)
    int64_t vec_size = n / 4;
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    float4* output_vec = reinterpret_cast<float4*>(output);

    // Grid-stride loop for vectorized processing
    for (int64_t i = tid; i < vec_size; i += stride) {
        float4 val = input_vec[i];
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        val.z = fmaxf(0.0f, val.z);
        val.w = fmaxf(0.0f, val.w);
        output_vec[i] = val;
    }

    // Handle remaining elements (tail processing)
    int64_t remaining_start = vec_size * 4;
    for (int64_t i = remaining_start + tid; i < n; i += stride) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

/**
 * @brief ReLU forward CUDA kernel (simple version for small tensors)
 *
 * Computes: output[i] = max(0, input[i])
 *
 * @param input Input tensor data
 * @param output Output tensor data
 * @param n Number of elements
 */
__global__ void relu_forward_kernel_simple(
    const float* input,
    float* output,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

/**
 * @brief ReLU CUDA kernel function
 *
 * Implements ReLU activation using hand-written CUDA kernels.
 * Automatically selects between vectorized and simple versions based on tensor size.
 *
 * Performance characteristics:
 * - Small tensors (< 1024 elements): Simple kernel with minimal overhead
 * - Large tensors: Vectorized kernel with float4 for better memory bandwidth
 *
 * @param ctx Kernel execution context
 */
void relu_cuda(KernelContext* ctx) {
    if (!ctx || !ctx->inputs || !ctx->outputs || !ctx->device_context) {
        return;
    }

    const auto& inputs = *ctx->inputs;
    auto& outputs = *ctx->outputs;
    if (inputs.empty() || outputs.empty() || !inputs[0] || !outputs[0]) {
        return;
    }

    // Get CUDA device context
    auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(
        ctx->device_context
    );
    if (!cuda_ctx) {
        return;
    }

    // Get input and output tensors
    const auto& input = inputs[0];
    auto& output = outputs[0];

    // Calculate total number of elements
    int64_t n = input->shape().numel();

    // Kernel launch configuration
    const int threads = 256;  // Optimal for most GPUs (multiple of warp size 32)

    // Choose kernel based on tensor size
    if (n < 1024) {
        // Small tensor: use simple kernel
        int blocks = (n + threads - 1) / threads;
        relu_forward_kernel_simple<<<blocks, threads, 0, cuda_ctx->stream()>>>(
            static_cast<const float*>(input->data()),
            static_cast<float*>(output->data()),
            n
        );
    } else {
        // Large tensor: use vectorized kernel
        // Limit blocks to avoid excessive grid size
        int blocks = std::min((n + threads * 4 - 1) / (threads * 4), (int64_t)2048);
        relu_forward_kernel_vectorized<<<blocks, threads, 0, cuda_ctx->stream()>>>(
            static_cast<const float*>(input->data()),
            static_cast<float*>(output->data()),
            n
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
 * @brief ReLU CUDA kernel registrar
 * 
 * Automatically registers the ReLU CUDA kernel on program startup.
 */
namespace {
    struct ReLUCUDARegistrar {
        ReLUCUDARegistrar() {
            KernelRegistry::instance().register_kernel(
                core::OpType::kRELU,
                core::DeviceType::CUDA,
                core::DataType::FLOAT32,
                relu_cuda
            );
        }
    };
    static ReLUCUDARegistrar g_relu_cuda_registrar;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace mini_infer






