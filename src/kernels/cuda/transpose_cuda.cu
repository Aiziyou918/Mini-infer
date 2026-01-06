#include "mini_infer/kernels/transpose.h"
#include "mini_infer/kernels/kernel_base.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace kernels {
namespace cuda {

/**
 * @brief Transpose CNHW to NCHW CUDA kernel
 *
 * Input layout (CNHW):  [C, N*H*W] where data is organized as [C][N0_hw, N1_hw, ...]
 * Output layout (NCHW): [N, C, H*W] where data is organized as [N][C0_hw, C1_hw, ...]
 *
 * Each thread handles one spatial element.
 */
__global__ void transpose_CNHW_to_NCHW_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int C, int N, int H, int W
) {
    int spatial_size = H * W;
    int total = N * C * spatial_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        // Decode output index: output[n, c, hw]
        int hw = i % spatial_size;
        int c = (i / spatial_size) % C;
        int n = i / (spatial_size * C);

        // Calculate input index: input[c, n * spatial_size + hw]
        int input_idx = c * N * spatial_size + n * spatial_size + hw;

        output[i] = input[input_idx];
    }
}

/**
 * @brief Transpose CNHW to NCHW CUDA implementation for float
 */
void transpose_CNHW_to_NCHW_cuda_impl_float(
    const float* input,
    float* output,
    int C, int N, int H, int W
) {
    int total = N * C * H * W;
    const int threads = 256;
    int blocks = std::min((total + threads - 1) / threads, 65535);

    cudaStream_t stream = nullptr;
    auto* ctx = get_current_device_context();
    if (ctx) {
        auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(ctx);
        stream = cuda_ctx->stream();
    }

    transpose_CNHW_to_NCHW_kernel<<<blocks, threads, 0, stream>>>(
        input, output, C, N, H, W
    );

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDA] Transpose kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief Transpose CNHW to NCHW CUDA kernel for int32_t
 */
__global__ void transpose_CNHW_to_NCHW_kernel_int32(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ output,
    int C, int N, int H, int W
) {
    int spatial_size = H * W;
    int total = N * C * spatial_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int hw = i % spatial_size;
        int c = (i / spatial_size) % C;
        int n = i / (spatial_size * C);

        int input_idx = c * N * spatial_size + n * spatial_size + hw;
        output[i] = input[input_idx];
    }
}

/**
 * @brief Transpose CNHW to NCHW CUDA implementation for int32_t
 */
void transpose_CNHW_to_NCHW_cuda_impl_int32(
    const int32_t* input,
    int32_t* output,
    int C, int N, int H, int W
) {
    int total = N * C * H * W;
    const int threads = 256;
    int blocks = std::min((total + threads - 1) / threads, 65535);

    cudaStream_t stream = nullptr;
    auto* ctx = get_current_device_context();
    if (ctx) {
        auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(ctx);
        stream = cuda_ctx->stream();
    }

    transpose_CNHW_to_NCHW_kernel_int32<<<blocks, threads, 0, stream>>>(
        input, output, C, N, H, W
    );

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDA] Transpose kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief Check if CUDA is available
 */
bool is_cuda_available_transpose() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

/**
 * @brief Transpose CUDA kernel registrar
 */
namespace {
    void register_transpose_cuda_kernels() {
        TransposeRegistry_CNHW_to_NCHW<float>::instance().register_kernel(
            KernelBackend::CUDA,
            transpose_CNHW_to_NCHW_cuda_impl_float,
            is_cuda_available_transpose,
            200  // Higher priority than CPU
        );

        TransposeRegistry_CNHW_to_NCHW<int32_t>::instance().register_kernel(
            KernelBackend::CUDA,
            transpose_CNHW_to_NCHW_cuda_impl_int32,
            is_cuda_available_transpose,
            200
        );
    }

    struct TransposeCUDARegistrar {
        TransposeCUDARegistrar() {
            register_transpose_cuda_kernels();
        }
    };
    static TransposeCUDARegistrar g_transpose_cuda_registrar;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace mini_infer
