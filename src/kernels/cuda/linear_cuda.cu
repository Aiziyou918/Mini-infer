#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/operators/linear.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace kernels {
namespace cuda {

// Tile size for shared memory blocking
constexpr int TILE_SIZE = 32;

/**
 * @brief GEMM_NT CUDA kernel (A * B^T)
 *
 * Computes: C = A * B^T
 * Where A: [M, K], B: [N, K] (stored as [N, K], transposed during computation), C: [M, N]
 *
 * This is the operation needed for Linear layer: output = input @ weight^T
 */
__global__ void linear_gemm_nt_kernel_tiled(
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
__global__ void linear_gemm_nt_kernel_simple(
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
 * @brief Bias addition kernel for Linear layer
 *
 * Adds bias to each row: output[i, j] += bias[j]
 */
__global__ void linear_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch_size,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;

    if (idx < total) {
        int j = idx % out_features;
        output[idx] += bias[j];
    }
}

/**
 * @brief Linear CUDA kernel function
 *
 * Implements Linear layer: output = input @ weight^T + bias
 */
void linear_cuda(KernelContext* ctx) {
    if (!ctx || !ctx->inputs || !ctx->outputs || !ctx->device_context) {
        return;
    }

    const auto& inputs = *ctx->inputs;
    auto& outputs = *ctx->outputs;
    const auto* param = ctx->param<operators::LinearParam>();
    if (!param) {
        return;
    }

    const size_t expected_inputs = param->use_bias ? 3 : 2;
    if (inputs.size() != expected_inputs || outputs.empty()) {
        return;
    }

    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    auto output = outputs[0];
    if (!input || !weight || !output) {
        return;
    }

    auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(ctx->device_context);
    if (!cuda_ctx) {
        return;
    }

    const auto& input_shape = input->shape();
    const auto& weight_shape = weight->shape();
    const auto& output_shape = output->shape();
    if (input_shape.ndim() < 2 || weight_shape.ndim() != 2 || output_shape.ndim() < 2) {
        return;
    }

    // Calculate batch size (product of all dimensions except last)
    int batch_size = 1;
    for (size_t i = 0; i + 1 < input_shape.ndim(); ++i) {
        batch_size *= static_cast<int>(input_shape[i]);
    }

    const int in_features = static_cast<int>(input_shape[input_shape.ndim() - 1]);
    const int out_features = static_cast<int>(output_shape[output_shape.ndim() - 1]);

    const float* input_data = static_cast<const float*>(input->data());
    const float* weight_data = static_cast<const float*>(weight->data());
    float* output_data = static_cast<float*>(output->data());

    // GEMM: output = input @ weight^T
    // input: [batch_size, in_features], weight: [out_features, in_features]
    // output: [batch_size, out_features]
    int M = batch_size;
    int K = in_features;
    int N = out_features;

    if (M * N < 4096) {
        // Small matrix: use simple kernel
        dim3 threads(16, 16);
        dim3 blocks((N + threads.x - 1) / threads.x,
                    (M + threads.y - 1) / threads.y);

        linear_gemm_nt_kernel_simple<<<blocks, threads, 0, cuda_ctx->stream()>>>(
            input_data, weight_data, output_data, M, K, N
        );
    } else {
        // Large matrix: use tiled kernel
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                    (M + TILE_SIZE - 1) / TILE_SIZE);

        linear_gemm_nt_kernel_tiled<<<blocks, threads, 0, cuda_ctx->stream()>>>(
            input_data, weight_data, output_data, M, K, N
        );
    }

    // Add bias if needed
    if (param->use_bias) {
        const float* bias_data = static_cast<const float*>(inputs[2]->data());
        int total = batch_size * out_features;
        const int threads = 256;
        int blocks = (total + threads - 1) / threads;

        linear_bias_kernel<<<blocks, threads, 0, cuda_ctx->stream()>>>(
            output_data, bias_data, batch_size, out_features
        );
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDA] Linear kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief Linear CUDA kernel registrar
 */
namespace {
    struct LinearCUDARegistrar {
        LinearCUDARegistrar() {
            KernelRegistry::instance().register_kernel(
                core::OpType::kGEMM,
                core::DeviceType::CUDA,
                core::DataType::FLOAT32,
                linear_cuda
            );
        }
    };
    static LinearCUDARegistrar g_linear_cuda_registrar;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace mini_infer
