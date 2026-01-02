#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/operators/softmax.h"
#include "mini_infer/utils/logger.h"

#include <string>
#include <cfloat>

namespace mini_infer {
namespace kernels {
namespace cuda {

/**
 * @brief Softmax CUDA kernel (numerically stable)
 *
 * Computes softmax along the specified axis using the max-subtraction trick
 * for numerical stability: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 *
 * Each block handles one "row" (outer * inner combination).
 * Uses shared memory for reduction operations.
 */
__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer,
    int dim,
    int inner
) {
    extern __shared__ float shared[];

    int row_idx = blockIdx.x;
    if (row_idx >= outer * inner) return;

    int o = row_idx / inner;
    int i = row_idx % inner;

    int tid = threadIdx.x;
    int base = o * dim * inner + i;

    // Step 1: Find max value (parallel reduction)
    float local_max = -FLT_MAX;
    for (int d = tid; d < dim; d += blockDim.x) {
        float val = input[base + d * inner];
        local_max = fmaxf(local_max, val);
    }

    // Store local max in shared memory
    shared[tid] = local_max;
    __syncthreads();

    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    float max_val = shared[0];
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float val = expf(input[base + d * inner] - max_val);
        output[base + d * inner] = val;  // Store exp temporarily
        local_sum += val;
    }

    // Store local sum in shared memory
    shared[tid] = local_sum;
    __syncthreads();

    // Reduce to find global sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float sum_val = shared[0];
    __syncthreads();

    // Step 3: Normalize
    float inv_sum = (sum_val > 0.0f) ? (1.0f / sum_val) : 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        output[base + d * inner] *= inv_sum;
    }
}

/**
 * @brief Simple softmax kernel for small dimensions
 *
 * Each thread handles one complete softmax computation.
 */
__global__ void softmax_kernel_simple(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer,
    int dim,
    int inner
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;

    if (idx >= total) return;

    int o = idx / inner;
    int i = idx % inner;
    int base = o * dim * inner + i;

    // Find max
    float max_val = -FLT_MAX;
    for (int d = 0; d < dim; ++d) {
        max_val = fmaxf(max_val, input[base + d * inner]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float val = expf(input[base + d * inner] - max_val);
        output[base + d * inner] = val;
        sum += val;
    }

    // Normalize
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int d = 0; d < dim; ++d) {
        output[base + d * inner] *= inv_sum;
    }
}

/**
 * @brief Softmax CUDA kernel function
 */
void softmax_cuda(KernelContext* ctx) {
    if (!ctx || !ctx->inputs || !ctx->outputs || !ctx->device_context) {
        return;
    }

    const auto& inputs = *ctx->inputs;
    auto& outputs = *ctx->outputs;
    if (inputs.empty() || outputs.empty() || !inputs[0] || !outputs[0]) {
        return;
    }

    const auto* param = ctx->param<operators::SoftmaxParam>();
    const auto& shape = inputs[0]->shape();
    if (!param || shape.ndim() == 0) {
        return;
    }

    auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(ctx->device_context);
    if (!cuda_ctx) {
        return;
    }

    const auto dims = shape.dims();
    int axis = param->axis;
    if (axis < 0) {
        axis += static_cast<int>(dims.size());
    }
    if (axis < 0 || axis >= static_cast<int>(dims.size())) {
        return;
    }

    // Calculate outer, dim, inner
    int outer = 1;
    for (int i = 0; i < axis; ++i) {
        outer *= static_cast<int>(dims[static_cast<size_t>(i)]);
    }
    int inner = 1;
    for (size_t i = static_cast<size_t>(axis + 1); i < dims.size(); ++i) {
        inner *= static_cast<int>(dims[i]);
    }
    int dim = static_cast<int>(dims[static_cast<size_t>(axis)]);

    const float* input = static_cast<const float*>(inputs[0]->data());
    float* output = static_cast<float*>(outputs[0]->data());

    int total_rows = outer * inner;

    // Choose kernel based on dimension size
    if (dim <= 32) {
        // Small dimension: use simple kernel
        const int threads = 256;
        int blocks = (total_rows + threads - 1) / threads;
        softmax_kernel_simple<<<blocks, threads, 0, cuda_ctx->stream()>>>(
            input, output, outer, dim, inner
        );
    } else {
        // Large dimension: use shared memory kernel
        int threads = std::min(256, ((dim + 31) / 32) * 32);  // Round up to multiple of 32
        int shared_size = threads * sizeof(float);
        softmax_kernel<<<total_rows, threads, shared_size, cuda_ctx->stream()>>>(
            input, output, outer, dim, inner
        );
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDA] Softmax kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief Softmax CUDA kernel registrar
 */
namespace {
    struct SoftmaxCUDARegistrar {
        SoftmaxCUDARegistrar() {
            KernelRegistry::instance().register_kernel(
                core::OpType::kSOFTMAX,
                core::DeviceType::CUDA,
                core::DataType::FLOAT32,
                softmax_cuda
            );
        }
    };
    static SoftmaxCUDARegistrar g_softmax_cuda_registrar;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace mini_infer
