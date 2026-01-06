#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace operators {

namespace {

constexpr int TILE_SIZE = 32;

/**
 * @brief GEMM_NT CUDA kernel (A * B^T) with tiling
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
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        int b_row = t * TILE_SIZE + ty;
        if (col < N && b_row < K) {
            Bs[ty][tx] = B[col * K + b_row];
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
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

/**
 * @brief Bias addition kernel
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

}  // namespace

/**
 * @brief Linear (Fully Connected) CUDA Plugin
 */
class LinearCUDAPlugin : public CUDAPlugin<LinearCUDAPlugin, LinearParam> {
public:
    LinearCUDAPlugin() {
        param_ = std::make_shared<LinearParam>();
    }
    ~LinearCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Gemm";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kGEMM;
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

        if (input_shape.ndim() < 1 || weight_shape.ndim() != 2) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        std::vector<int64_t> output_dims;
        for (size_t i = 0; i + 1 < input_shape.ndim(); ++i) {
            output_dims.push_back(input_shape[i]);
        }
        output_dims.push_back(weight_shape[0]);

        output_shapes.clear();
        output_shapes.push_back(core::Shape(output_dims));
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {

        const size_t expected_inputs = (param_ && param_->use_bias) ? 3 : 2;
        if (inputs.size() != expected_inputs || outputs.size() != 1) {
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

        const auto& input_shape = input->shape();
        const auto& output_shape = output->shape();

        int batch_size = 1;
        for (size_t i = 0; i + 1 < input_shape.ndim(); ++i) {
            batch_size *= static_cast<int>(input_shape[i]);
        }

        const int in_features = static_cast<int>(input_shape[input_shape.ndim() - 1]);
        const int out_features = static_cast<int>(output_shape[output_shape.ndim() - 1]);

        const float* input_data = static_cast<const float*>(input->data());
        const float* weight_data = static_cast<const float*>(weight->data());
        float* output_data = static_cast<float*>(output->data());

        int M = batch_size;
        int K = in_features;
        int N = out_features;

        if (M * N < 4096) {
            dim3 threads(16, 16);
            dim3 blocks((N + threads.x - 1) / threads.x,
                        (M + threads.y - 1) / threads.y);
            linear_gemm_nt_kernel_simple<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                input_data, weight_data, output_data, M, K, N
            );
        } else {
            dim3 threads(TILE_SIZE, TILE_SIZE);
            dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                        (M + TILE_SIZE - 1) / TILE_SIZE);
            linear_gemm_nt_kernel_tiled<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                input_data, weight_data, output_data, M, K, N
            );
        }

        if (param_ && param_->use_bias && inputs.size() > 2 && inputs[2]) {
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
            MI_LOG_ERROR("[CUDA] Linear plugin error: " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }
};

// Define creator and register plugin
REGISTER_PLUGIN_SIMPLE(LinearCUDAPlugin, "Gemm", kGEMM, CUDA)

}  // namespace operators
}  // namespace mini_infer
