#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace kernels {
namespace cuda {

/**
 * @brief Memory copy kernel (vectorized)
 *
 * Uses float4 for better memory bandwidth.
 */
__global__ void memcpy_kernel_vectorized(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t n
) {
    int64_t vec_size = n / 4;
    const float4* src_vec = reinterpret_cast<const float4*>(src);
    float4* dst_vec = reinterpret_cast<float4*>(dst);

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    // Vectorized copy
    for (int64_t i = idx; i < vec_size; i += stride) {
        dst_vec[i] = src_vec[i];
    }

    // Handle remaining elements
    int64_t remaining_start = vec_size * 4;
    for (int64_t i = remaining_start + idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

/**
 * @brief Simple memory copy kernel
 */
__global__ void memcpy_kernel_simple(
    const char* __restrict__ src,
    char* __restrict__ dst,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

/**
 * @brief Flatten CUDA kernel function
 *
 * Flatten is essentially a memory copy with shape change.
 * The data layout remains the same, only the shape interpretation changes.
 */
void flatten_cuda(KernelContext* ctx) {
    if (!ctx || !ctx->inputs || !ctx->outputs || !ctx->device_context) {
        return;
    }

    const auto& inputs = *ctx->inputs;
    auto& outputs = *ctx->outputs;
    if (inputs.empty() || outputs.empty() || !inputs[0] || !outputs[0]) {
        return;
    }

    auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(ctx->device_context);
    if (!cuda_ctx) {
        return;
    }

    const void* src = inputs[0]->data();
    void* dst = outputs[0]->data();
    size_t size_bytes = inputs[0]->size_in_bytes();

    // Skip if same memory location
    if (src == dst || !src || !dst) {
        return;
    }

    // Use cudaMemcpyAsync for best performance
    cudaMemcpyAsync(dst, src, size_bytes, cudaMemcpyDeviceToDevice, cuda_ctx->stream());

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDA] Flatten kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief Register flatten kernel for a specific data type
 */
void register_flatten_dtype_cuda(core::DataType dtype) {
    KernelRegistry::instance().register_kernel(
        core::OpType::kFLATTEN,
        core::DeviceType::CUDA,
        dtype,
        flatten_cuda
    );
}

/**
 * @brief Flatten CUDA kernel registrar
 */
namespace {
    struct FlattenCUDARegistrar {
        FlattenCUDARegistrar() {
            register_flatten_dtype_cuda(core::DataType::FLOAT32);
            register_flatten_dtype_cuda(core::DataType::FLOAT16);
            register_flatten_dtype_cuda(core::DataType::INT32);
            register_flatten_dtype_cuda(core::DataType::INT64);
            register_flatten_dtype_cuda(core::DataType::INT8);
            register_flatten_dtype_cuda(core::DataType::UINT8);
            register_flatten_dtype_cuda(core::DataType::BOOL);
        }
    };
    static FlattenCUDARegistrar g_flatten_cuda_registrar;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace mini_infer
