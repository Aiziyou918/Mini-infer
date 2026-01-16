#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/kernels/gemm.h"

#include <algorithm>

namespace mini_infer {
namespace operators {

/**
 * @brief MatMul CPU Plugin
 *
 * Performs matrix multiplication following ONNX MatMul semantics:
 * - 2D: [M, K] @ [K, N] -> [M, N]
 * - 3D+: Batch dimensions are broadcast, last two dims do matmul
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

        const auto& a_shape = input_shapes[0];
        const auto& b_shape = input_shapes[1];

        if (a_shape.ndim() < 1 || b_shape.ndim() < 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Handle 1D cases
        bool a_is_1d = (a_shape.ndim() == 1);
        bool b_is_1d = (b_shape.ndim() == 1);

        std::vector<int64_t> a_dims = a_shape.dims();
        std::vector<int64_t> b_dims = b_shape.dims();

        // Prepend 1 to A if 1D
        if (a_is_1d) {
            a_dims.insert(a_dims.begin(), 1);
        }
        // Append 1 to B if 1D
        if (b_is_1d) {
            b_dims.push_back(1);
        }

        const size_t a_ndim = a_dims.size();
        const size_t b_ndim = b_dims.size();

        // Get matrix dimensions
        const int64_t M = a_dims[a_ndim - 2];
        const int64_t K_a = a_dims[a_ndim - 1];
        const int64_t K_b = b_dims[b_ndim - 2];
        const int64_t N = b_dims[b_ndim - 1];

        if (K_a != K_b) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Broadcast batch dimensions
        const size_t max_batch_ndim = std::max(a_ndim, b_ndim) - 2;
        std::vector<int64_t> output_dims;

        for (size_t i = 0; i < max_batch_ndim; ++i) {
            // Align batch dimensions from the right
            int64_t a_dim = 1;
            int64_t b_dim = 1;

            if (i >= max_batch_ndim - (a_ndim - 2)) {
                a_dim = a_dims[i - (max_batch_ndim - (a_ndim - 2))];
            }
            if (i >= max_batch_ndim - (b_ndim - 2)) {
                b_dim = b_dims[i - (max_batch_ndim - (b_ndim - 2))];
            }

            if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }

            output_dims.push_back(std::max(a_dim, b_dim));
        }

        // Add matrix dimensions
        if (!a_is_1d) {
            output_dims.push_back(M);
        }
        if (!b_is_1d) {
            output_dims.push_back(N);
        }

        // Handle scalar output case
        if (output_dims.empty()) {
            output_dims.push_back(1);
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(output_dims));
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

        const auto& a = inputs[0];
        const auto& b = inputs[1];
        auto& out = outputs[0];

        if (!a || !b || !out) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        if (a->dtype() != core::DataType::FLOAT32 || b->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const core::Shape& a_shape = a->shape();
        const core::Shape& b_shape = b->shape();
        const core::Shape& out_shape = out->shape();

        // Handle 1D cases
        bool a_is_1d = (a_shape.ndim() == 1);
        bool b_is_1d = (b_shape.ndim() == 1);

        std::vector<int64_t> a_dims = a_shape.dims();
        std::vector<int64_t> b_dims = b_shape.dims();

        if (a_is_1d) {
            a_dims.insert(a_dims.begin(), 1);
        }
        if (b_is_1d) {
            b_dims.push_back(1);
        }

        const size_t a_ndim = a_dims.size();
        const size_t b_ndim = b_dims.size();

        const int M = static_cast<int>(a_dims[a_ndim - 2]);
        const int K = static_cast<int>(a_dims[a_ndim - 1]);
        const int N = static_cast<int>(b_dims[b_ndim - 1]);

        const float* a_data = static_cast<const float*>(a->data());
        const float* b_data = static_cast<const float*>(b->data());
        float* out_data = static_cast<float*>(out->data());

        // Handle simple 2D case
        if (a_shape.ndim() <= 2 && b_shape.ndim() <= 2) {
            // Simple 2D matmul: C = A @ B (no transpose)
            kernels::GEMMKernel::gemm_nn<float>(
                a_data, b_data, out_data,
                M, N, K,
                kernels::KernelBackend::CPU
            );
            return core::Status::SUCCESS;
        }

        // Batched matmul with proper broadcasting
        const int64_t a_batch_stride = M * K;
        const int64_t b_batch_stride = K * N;
        const int64_t out_batch_stride = M * N;

        // Get batch dimensions (excluding last 2 matrix dims)
        const size_t max_batch_ndim = std::max(a_ndim, b_ndim) - 2;

        // Build broadcast batch dimensions for A and B
        // Align from the right (before matrix dims)
        std::vector<int64_t> a_batch_dims(max_batch_ndim, 1);
        std::vector<int64_t> b_batch_dims(max_batch_ndim, 1);
        std::vector<int64_t> out_batch_dims(max_batch_ndim);

        for (size_t i = 0; i < max_batch_ndim; ++i) {
            // Index from the left in the aligned batch dimensions
            if (i >= max_batch_ndim - (a_ndim - 2)) {
                a_batch_dims[i] = a_dims[i - (max_batch_ndim - (a_ndim - 2))];
            }
            if (i >= max_batch_ndim - (b_ndim - 2)) {
                b_batch_dims[i] = b_dims[i - (max_batch_ndim - (b_ndim - 2))];
            }
            out_batch_dims[i] = std::max(a_batch_dims[i], b_batch_dims[i]);
        }

        // Compute output batch strides (for converting flat index to multi-dim index)
        std::vector<int64_t> out_batch_strides(max_batch_ndim);
        int64_t stride = 1;
        for (int i = static_cast<int>(max_batch_ndim) - 1; i >= 0; --i) {
            out_batch_strides[i] = stride;
            stride *= out_batch_dims[i];
        }
        const int64_t total_batches = stride;

        // Compute input batch strides for A and B
        std::vector<int64_t> a_batch_strides(max_batch_ndim);
        stride = 1;
        for (int i = static_cast<int>(max_batch_ndim) - 1; i >= 0; --i) {
            // If dimension is 1 (broadcast), stride is 0
            a_batch_strides[i] = (a_batch_dims[i] == 1) ? 0 : stride;
            stride *= a_batch_dims[i];
        }

        std::vector<int64_t> b_batch_strides(max_batch_ndim);
        stride = 1;
        for (int i = static_cast<int>(max_batch_ndim) - 1; i >= 0; --i) {
            b_batch_strides[i] = (b_batch_dims[i] == 1) ? 0 : stride;
            stride *= b_batch_dims[i];
        }

        // Process each batch
        for (int64_t batch = 0; batch < total_batches; ++batch) {
            // Convert flat batch index to multi-dimensional indices
            // Then compute the corresponding A and B batch indices
            int64_t a_batch_idx = 0;
            int64_t b_batch_idx = 0;
            int64_t remaining = batch;

            for (size_t d = 0; d < max_batch_ndim; ++d) {
                int64_t idx = remaining / out_batch_strides[d];
                remaining = remaining % out_batch_strides[d];

                // For A: if this dimension is broadcast (size 1), don't advance
                a_batch_idx += idx * a_batch_strides[d];
                // For B: if this dimension is broadcast (size 1), don't advance
                b_batch_idx += idx * b_batch_strides[d];
            }

            const float* a_ptr = a_data + a_batch_idx * a_batch_stride;
            const float* b_ptr = b_data + b_batch_idx * b_batch_stride;
            float* out_ptr = out_data + batch * out_batch_stride;

            kernels::GEMMKernel::gemm_nn<float>(
                a_ptr, b_ptr, out_ptr,
                M, N, K,
                kernels::KernelBackend::CPU
            );
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(MatMulCPUPlugin, "MatMul", kMATMUL, CPU)

}  // namespace operators
}  // namespace mini_infer
