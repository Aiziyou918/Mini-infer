#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace kernels {
namespace cuda {

/**
 * @brief Reshape CUDA kernel function
 *
 * Reshape is essentially a memory copy with shape change.
 * The data layout remains the same, only the shape interpretation changes.
 */
void reshape_cuda(KernelContext* ctx) {
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
        MI_LOG_ERROR("[CUDA] Reshape kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief Register reshape kernel for a specific data type
 */
void register_reshape_dtype_cuda(core::DataType dtype) {
    KernelRegistry::instance().register_kernel(
        core::OpType::kRESHAPE,
        core::DeviceType::CUDA,
        dtype,
        reshape_cuda
    );
}

/**
 * @brief Reshape CUDA kernel registrar
 */
namespace {
    struct ReshapeCUDARegistrar {
        ReshapeCUDARegistrar() {
            register_reshape_dtype_cuda(core::DataType::FLOAT32);
            register_reshape_dtype_cuda(core::DataType::FLOAT16);
            register_reshape_dtype_cuda(core::DataType::INT32);
            register_reshape_dtype_cuda(core::DataType::INT64);
            register_reshape_dtype_cuda(core::DataType::INT8);
            register_reshape_dtype_cuda(core::DataType::UINT8);
            register_reshape_dtype_cuda(core::DataType::BOOL);
        }
    };
    static ReshapeCUDARegistrar g_reshape_cuda_registrar;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace mini_infer
