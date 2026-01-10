#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <algorithm>
#include <string>

namespace mini_infer {
namespace operators {

namespace {

// Batch MatMul kernel: C[b] = A[b] @ B[b]
__global__ void batched_matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int64_t a_batch, int64_t b_batch, int64_t c_batch,
    int64_t a_mat_size, int64_t b_mat_size, int64_t c_mat_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = c_batch * M * N;
    if (idx >= total) return;

    int64_t batch = idx / (M * N);
    int64_t mat_idx = idx % (M * N);
    int m = mat_idx / N;
    int n = mat_idx % N;

    int64_t a_idx = (a_batch == 1) ? 0 : batch;
    int64_t b_idx = (b_batch == 1) ? 0 : batch;

    const float* a_ptr = A + a_idx * a_mat_size + m * K;
    const float* b_ptr = B + b_idx * b_mat_size + n;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += a_ptr[k] * b_ptr[k * N];
    }

    C[idx] = sum;
}

}  // namespace

/**
 * @brief MatMul CUDA Plugin
 *
 * Implements batch matrix multiplication on GPU.
 */
class MatMulCUDAPlugin : public SimpleCUDAPlugin<MatMulCUDAPlugin> {
public:
    MatMulCUDAPlugin() = default;
    ~MatMulCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "MatMul";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kMATMUL;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return 2;
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {
        if (input_shapes.size() != 2) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& a_dims = input_shapes[0].dims();
        const auto& b_dims = input_shapes[1].dims();

        if (a_dims.size() < 2 || b_dims.size() < 2) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        int64_t M = a_dims[a_dims.size() - 2];
        int64_t K_a = a_dims[a_dims.size() - 1];
        int64_t K_b = b_dims[b_dims.size() - 2];
        int64_t N = b_dims[b_dims.size() - 1];

        if (K_a != K_b) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        size_t a_batch_ndim = a_dims.size() - 2;
        size_t b_batch_ndim = b_dims.size() - 2;
        size_t out_batch_ndim = std::max(a_batch_ndim, b_batch_ndim);

        std::vector<int64_t> out_dims;
        for (size_t i = 0; i < out_batch_ndim; ++i) {
            int64_t a_dim = 1, b_dim = 1;
            if (i >= out_batch_ndim - a_batch_ndim) {
                a_dim = a_dims[i - (out_batch_ndim - a_batch_ndim)];
            }
            if (i >= out_batch_ndim - b_batch_ndim) {
                b_dim = b_dims[i - (out_batch_ndim - b_batch_ndim)];
            }

            if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
            out_dims.push_back(std::max(a_dim, b_dim));
        }

        out_dims.push_back(M);
        out_dims.push_back(N);

        output_shapes.clear();
        output_shapes.push_back(core::Shape(out_dims));
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {

        if (inputs.size() != 2 || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& A = inputs[0];
        const auto& B = inputs[1];
        auto& C = outputs[0];

        if (!A || !B || !C || !context.device_context) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(
            context.device_context
        );
        if (!cuda_ctx) {
            return core::Status::ERROR_BACKEND;
        }

        if (A->dtype() != core::DataType::FLOAT32 ||
            B->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& a_dims = A->shape().dims();
        const auto& b_dims = B->shape().dims();
        const auto& c_dims = C->shape().dims();

        const int M = static_cast<int>(a_dims[a_dims.size() - 2]);
        const int K = static_cast<int>(a_dims[a_dims.size() - 1]);
        const int N = static_cast<int>(b_dims[b_dims.size() - 1]);

        int64_t a_batch = 1, b_batch = 1, c_batch = 1;
        for (size_t i = 0; i + 2 < a_dims.size(); ++i) {
            a_batch *= a_dims[i];
        }
        for (size_t i = 0; i + 2 < b_dims.size(); ++i) {
            b_batch *= b_dims[i];
        }
        for (size_t i = 0; i + 2 < c_dims.size(); ++i) {
            c_batch *= c_dims[i];
        }

        const int64_t a_mat_size = M * K;
        const int64_t b_mat_size = K * N;
        const int64_t c_mat_size = M * N;

        const int64_t total = c_batch * M * N;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;

        batched_matmul_kernel<<<blocks, threads, 0, cuda_ctx->stream()>>>(
            static_cast<const float*>(A->data()),
            static_cast<const float*>(B->data()),
            static_cast<float*>(C->data()),
            M, N, K,
            a_batch, b_batch, c_batch,
            a_mat_size, b_mat_size, c_mat_size
        );

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(MatMulCUDAPlugin, "MatMul", kMATMUL, CUDA)

}  // namespace operators
}  // namespace mini_infer
