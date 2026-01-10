#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>
#include <cstring>

namespace mini_infer {
namespace operators {

namespace {

// Simple GEMM for batch matmul: C = A @ B
void gemm_nn(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

}  // namespace

/**
 * @brief MatMul CPU Plugin
 *
 * Implements batch matrix multiplication.
 * Supports broadcasting for batch dimensions.
 * C = A @ B where A is [..., M, K] and B is [..., K, N]
 */
class MatMulCPUPlugin : public SimpleCPUPlugin<MatMulCPUPlugin> {
public:
    MatMulCPUPlugin() = default;
    ~MatMulCPUPlugin() override = default;

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

        // Get matrix dimensions
        int64_t M = a_dims[a_dims.size() - 2];
        int64_t K_a = a_dims[a_dims.size() - 1];
        int64_t K_b = b_dims[b_dims.size() - 2];
        int64_t N = b_dims[b_dims.size() - 1];

        // Check K dimension compatibility (allow dynamic)
        if (K_a != K_b && K_a != -1 && K_b != -1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Compute broadcast batch dimensions
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

            // Handle dynamic dimensions
            if (a_dim == -1 || b_dim == -1) {
                if (a_dim == 1) {
                    out_dims.push_back(b_dim);
                } else if (b_dim == 1) {
                    out_dims.push_back(a_dim);
                } else {
                    out_dims.push_back(-1);
                }
            } else if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            } else {
                out_dims.push_back(std::max(a_dim, b_dim));
            }
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
        (void)context;

        if (inputs.size() != 2 || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& A = inputs[0];
        const auto& B = inputs[1];
        auto& C = outputs[0];

        if (!A || !B || !C) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        if (A->dtype() != core::DataType::FLOAT32 ||
            B->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& a_dims = A->shape().dims();
        const auto& b_dims = B->shape().dims();
        const auto& c_dims = C->shape().dims();

        const int64_t M = a_dims[a_dims.size() - 2];
        const int64_t K = a_dims[a_dims.size() - 1];
        const int64_t N = b_dims[b_dims.size() - 1];

        // Compute batch sizes
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

        const float* a_data = static_cast<const float*>(A->data());
        const float* b_data = static_cast<const float*>(B->data());
        float* c_data = static_cast<float*>(C->data());

        const int64_t a_mat_size = M * K;
        const int64_t b_mat_size = K * N;
        const int64_t c_mat_size = M * N;

        // Batch matmul with broadcasting
        for (int64_t batch = 0; batch < c_batch; ++batch) {
            int64_t a_idx = (a_batch == 1) ? 0 : batch;
            int64_t b_idx = (b_batch == 1) ? 0 : batch;

            const float* a_ptr = a_data + a_idx * a_mat_size;
            const float* b_ptr = b_data + b_idx * b_mat_size;
            float* c_ptr = c_data + batch * c_mat_size;

            gemm_nn(a_ptr, b_ptr, c_ptr,
                    static_cast<int>(M),
                    static_cast<int>(N),
                    static_cast<int>(K));
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(MatMulCPUPlugin, "MatMul", kMATMUL, CPU)

}  // namespace operators
}  // namespace mini_infer
