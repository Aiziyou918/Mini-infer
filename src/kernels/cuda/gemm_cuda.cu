#include "mini_infer/kernels/gemm.h"
#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace kernels {
namespace cuda {

// Tile size for shared memory blocking
// 32x32 is optimal for most modern GPUs (fits in shared memory, good occupancy)
constexpr int TILE_SIZE = 32;

/**
 * @brief GEMM CUDA kernel using shared memory tiling
 *
 * Computes: C = A * B
 * Where A: [M, K], B: [K, N], C: [M, N]
 *
 * Performance optimizations:
 * - Shared memory tiling to reduce global memory access
 * - Coalesced memory access pattern
 * - Register blocking for better instruction-level parallelism
 * - Each thread computes one element of C
 *
 * Algorithm:
 * 1. Divide matrices into tiles of size TILE_SIZE x TILE_SIZE
 * 2. Load tiles from A and B into shared memory cooperatively
 * 3. Compute partial dot product using shared memory
 * 4. Accumulate results across all tiles
 *
 * @param A Input matrix A [M, K] in row-major order
 * @param B Input matrix B [K, N] in row-major order
 * @param C Output matrix C [M, N] in row-major order
 Number of rows in A and C
 * @param K Number of columns in A and rows in B
 * @param N Number of columns in B and C
 */
__global__ void gemm_kernel_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column indices for C
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Accumulator for dot product
    float sum = 0.0f;

    // Loop over tiles of A and B required to compute C element
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A into shared memory
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * @brief Simple GEMM kernel for small matrices (no tiling)
 *
 * Used for small matrices where shared memory overhead is not beneficial.
 * Each thread computes one element of C.
 *
 * @param A Input matrix A [M, K]
 * @param B Input matrix B [K, N]
 * @param C Output matrix C [M, N]
 * @param M Number of rows in A and C
 * @param K Number of columns in A and rows in B
 * @param N Number of columns in B and C
 */
__global__ void gemm_kernel_simple(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * @brief GEMM_NT CUDA kernel (A * B^T) using shared memory tiling
 *
 * Computes: C = A * B^T
 * Where A: [M, K], B: [N, K] (stored as [N, K]), C: [M, N]
 */
__global__ void gemm_nt_kernel_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A into shared memory
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B^T into shared memory
        // B is [N, K], we want B^T[k, n] = B[n, k]
        int b_row = t * TILE_SIZE + ty;  // k index
        if (col < N && b_row < K) {
            Bs[ty][tx] = B[col * K + b_row];  // B[col, b_row] = B^T[b_row, col]
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * @brief Simple GEMM_NT kernel for small matrices
 */
__global__ void gemm_nt_kernel_simple(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];  // B^T[k, col] = B[col, k]
        }
        C[row * N + col] = sum;
    }
}

/**
 * @brief GEMM_NT CUDA implementation for template registry
 *
 * This function is registered to GEMMRegistry_NT for use by linear layer etc.
 */
void gemm_nt_cuda_impl(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    cudaStream_t stream = nullptr;
    auto* ctx = get_current_device_context();
    if (ctx) {
        auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(ctx);
        stream = cuda_ctx->stream();
    }

    // Choose kernel based on matrix size
    if (M * N < 4096) {
        dim3 threads(16, 16);
        dim3 blocks((N + threads.x - 1) / threads.x,
                    (M + threads.y - 1) / threads.y);

        gemm_nt_kernel_simple<<<blocks, threads, 0, stream>>>(
            A, B, C, M, K, N
        );
    } else {
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                    (M + TILE_SIZE - 1) / TILE_SIZE);

        gemm_nt_kernel_tiled<<<blocks, threads, 0, stream>>>(
            A, B, C, M, K, N
        );
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDA] GEMM_NT kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief GEMM_NN CUDA implementation for template registry
 *
 * This function is registered to GEMMRegistry_NN for use by conv2d and other
 * operations that need GEMM_NN (C = A * B).
 */
void gemm_nn_cuda_impl(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    cudaStream_t stream = nullptr;
    auto* ctx = get_current_device_context();
    if (ctx) {
        auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(ctx);
        stream = cuda_ctx->stream();
    }

    // Choose kernel based on matrix size
    if (M * N < 4096) {
        dim3 threads(16, 16);
        dim3 blocks((N + threads.x - 1) / threads.x,
                    (M + threads.y - 1) / threads.y);

        gemm_kernel_simple<<<blocks, threads, 0, stream>>>(
            A, B, C, M, K, N
        );
    } else {
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                    (M + TILE_SIZE - 1) / TILE_SIZE);

        gemm_kernel_tiled<<<blocks, threads, 0, stream>>>(
            A, B, C, M, K, N
        );
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDA] GEMM_NN kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief Check if CUDA is available
 */
bool is_cuda_available_gemm() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

/**
 * @brief GEMM CUDA kernel registrar
 *
 * Registers GEMM_NN and GEMM_NT to the template-based registry.
 * Note: Linear layer uses linear_cuda.cu which implements the full Linear operation.
 */
namespace {
    struct GEMMCUDARegistrar {
        GEMMCUDARegistrar() {
            // Register to template-based registry for GEMM_NN
            GEMMRegistry_NN<float>::instance().register_kernel(
                KernelBackend::CUDA,
                gemm_nn_cuda_impl,
                is_cuda_available_gemm,
                200  // Higher priority than CPU
            );

            // Register to template-based registry for GEMM_NT
            GEMMRegistry_NT<float>::instance().register_kernel(
                KernelBackend::CUDA,
                gemm_nt_cuda_impl,
                is_cuda_available_gemm,
                200  // Higher priority than CPU
            );
        }
    };
    static GEMMCUDARegistrar g_gemm_cuda_registrar;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace mini_infer







