#include "mini_infer/kernels/bias.h"
#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace kernels {
namespace cuda {

/**
 * @brief Bias addition CUDA kernel (vectorized)
 *
 * Adds channel-wise bias to tensor: output[b, c, s] += bias[c]
 * Uses float4 vectorization for better memory bandwidth.
 *
 * @param output Output tensor data [batch_size, channels, spatial_size]
 * @param bias Bias data [channels]
 * @param batch_size Number of batches
 * @param channels Number of channels
 * @param spatial_size Spatial dimension size (H*W)
 */
__global__ void bias_add_kernel_vectorized(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch_size,
    int channels,
    int spatial_size
) {
    int total_elements = batch_size * channels * spatial_size;
    int vec_size = total_elements / 4;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float4* output_vec = reinterpret_cast<float4*>(output);

    // Vectorized processing
    for (int i = idx; i < vec_size; i += stride) {
        int base_idx = i * 4;
        float4 val = output_vec[i];

        // Calculate channel indices for each element
        int idx0 = base_idx;
        int idx1 = base_idx + 1;
        int idx2 = base_idx + 2;
        int idx3 = base_idx + 3;

        int c0 = (idx0 / spatial_size) % channels;
        int c1 = (idx1 / spatial_size) % channels;
        int c2 = (idx2 / spatial_size) % channels;
        int c3 = (idx3 / spatial_size) % channels;

        val.x += bias[c0];
        val.y += bias[c1];
        val.z += bias[c2];
        val.w += bias[c3];

        output_vec[i] = val;
    }

    // Handle remaining elements
    int remaining_start = vec_size * 4;
    for (int i = remaining_start + idx; i < total_elements; i += stride) {
        int c = (i / spatial_size) % channels;
        output[i] += bias[c];
    }
}

/**
 * @brief Bias addition CUDA kernel (simple version)
 *
 * Simple version for small tensors or non-aligned data.
 */
__global__ void bias_add_kernel_simple(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch_size,
    int channels,
    int spatial_size
) {
    int total_elements = batch_size * channels * spatial_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        int c = (i / spatial_size) % channels;
        output[i] += bias[c];
    }
}

/**
 * @brief Bias CUDA implementation for float
 */
void bias_cuda_impl_float(
    float* output,
    const float* bias,
    int batch_size,
    int channels,
    int spatial_size
) {
    int total_elements = batch_size * channels * spatial_size;

    const int threads = 256;
    int blocks = std::min((total_elements + threads - 1) / threads, 65535);

    // Get current CUDA stream
    cudaStream_t stream = nullptr;
    auto* ctx = get_current_device_context();
    if (ctx) {
        auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(ctx);
        stream = cuda_ctx->stream();
    }

    // Choose kernel based on size and alignment
    if (total_elements >= 1024 && (reinterpret_cast<uintptr_t>(output) % 16 == 0)) {
        bias_add_kernel_vectorized<<<blocks, threads, 0, stream>>>(
            output, bias, batch_size, channels, spatial_size
        );
    } else {
        bias_add_kernel_simple<<<blocks, threads, 0, stream>>>(
            output, bias, batch_size, channels, spatial_size
        );
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDA] Bias kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief Bias CUDA kernel for int32_t
 */
__global__ void bias_add_kernel_int32(
    int32_t* __restrict__ output,
    const int32_t* __restrict__ bias,
    int batch_size,
    int channels,
    int spatial_size
) {
    int total_elements = batch_size * channels * spatial_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        int c = (i / spatial_size) % channels;
        output[i] += bias[c];
    }
}

/**
 * @brief Bias CUDA implementation for int32_t
 */
void bias_cuda_impl_int32(
    int32_t* output,
    const int32_t* bias,
    int batch_size,
    int channels,
    int spatial_size
) {
    int total_elements = batch_size * channels * spatial_size;

    const int threads = 256;
    int blocks = std::min((total_elements + threads - 1) / threads, 65535);

    cudaStream_t stream = nullptr;
    auto* ctx = get_current_device_context();
    if (ctx) {
        auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(ctx);
        stream = cuda_ctx->stream();
    }

    bias_add_kernel_int32<<<blocks, threads, 0, stream>>>(
        output, bias, batch_size, channels, spatial_size
    );

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDA] Bias kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief Check if CUDA is available
 */
bool is_cuda_available_bias() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

/**
 * @brief Bias CUDA kernel registrar
 */
namespace {
    void register_bias_cuda_kernels() {
        BiasRegistry<float>::instance().register_kernel(
            KernelBackend::CUDA,
            bias_cuda_impl_float,
            is_cuda_available_bias,
            200  // Higher priority than CPU
        );

        BiasRegistry<int32_t>::instance().register_kernel(
            KernelBackend::CUDA,
            bias_cuda_impl_int32,
            is_cuda_available_bias,
            200
        );
    }

    struct BiasCUDARegistrar {
        BiasCUDARegistrar() {
            register_bias_cuda_kernels();
        }
    };
    static BiasCUDARegistrar g_bias_cuda_registrar;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace mini_infer
