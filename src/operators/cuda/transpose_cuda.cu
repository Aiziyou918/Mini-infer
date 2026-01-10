#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <algorithm>
#include <string>
#include <vector>

namespace mini_infer {
namespace operators {

namespace {

// Transpose kernel for general permutation
__global__ void transpose_kernel(
    const float* input,
    float* output,
    const int64_t* in_strides,
    const int64_t* out_strides,
    const int64_t* perm,
    int ndim,
    int64_t total
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total) return;

    // Convert output linear index to coordinates
    int64_t in_idx = 0;
    int64_t remaining = out_idx;

    for (int d = 0; d < ndim; ++d) {
        int64_t coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        in_idx += coord * in_strides[perm[d]];
    }

    output[out_idx] = input[in_idx];
}

}  // namespace

/**
 * @brief Transpose CUDA Plugin
 */
class TransposeCUDAPlugin : public CUDAPlugin<TransposeCUDAPlugin, TransposeParam> {
public:
    TransposeCUDAPlugin() {
        param_ = std::make_shared<TransposeParam>();
    }
    ~TransposeCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Transpose";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kTRANSPOSE;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return 1;
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {
        if (input_shapes.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& in_shape = input_shapes[0];
        const auto& in_dims = in_shape.dims();
        size_t ndim = in_dims.size();

        std::vector<int64_t> perm;
        if (param_ && !param_->perm.empty()) {
            perm = param_->perm;
        } else {
            perm.resize(ndim);
            for (size_t i = 0; i < ndim; ++i) {
                perm[i] = static_cast<int64_t>(ndim - 1 - i);
            }
        }

        if (perm.size() != ndim) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        std::vector<int64_t> out_dims(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            int64_t axis = perm[i];
            if (axis < 0) axis += static_cast<int64_t>(ndim);
            if (axis < 0 || axis >= static_cast<int64_t>(ndim)) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
            out_dims[i] = in_dims[axis];
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(out_dims));
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {

        if (inputs.size() != 1 || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input = inputs[0];
        auto& output = outputs[0];

        if (!input || !output || !context.device_context) {
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

        const auto& in_shape = input->shape();
        const auto& in_dims = in_shape.dims();
        size_t ndim = in_dims.size();

        // Get permutation
        std::vector<int64_t> perm;
        if (param_ && !param_->perm.empty()) {
            perm = param_->perm;
        } else {
            perm.resize(ndim);
            for (size_t i = 0; i < ndim; ++i) {
                perm[i] = static_cast<int64_t>(ndim - 1 - i);
            }
        }

        // Normalize negative axes
        for (auto& p : perm) {
            if (p < 0) p += static_cast<int64_t>(ndim);
        }

        // Compute input strides
        std::vector<int64_t> in_strides(ndim);
        int64_t stride = 1;
        for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
            in_strides[i] = stride;
            stride *= in_dims[i];
        }

        // Compute output strides
        const auto& out_dims = output->shape().dims();
        std::vector<int64_t> out_strides(ndim);
        stride = 1;
        for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
            out_strides[i] = stride;
            stride *= out_dims[i];
        }

        // Allocate device memory for strides and perm
        int64_t* d_in_strides;
        int64_t* d_out_strides;
        int64_t* d_perm;

        cudaMalloc(&d_in_strides, ndim * sizeof(int64_t));
        cudaMalloc(&d_out_strides, ndim * sizeof(int64_t));
        cudaMalloc(&d_perm, ndim * sizeof(int64_t));

        cudaMemcpyAsync(d_in_strides, in_strides.data(), ndim * sizeof(int64_t),
                        cudaMemcpyHostToDevice, cuda_ctx->stream());
        cudaMemcpyAsync(d_out_strides, out_strides.data(), ndim * sizeof(int64_t),
                        cudaMemcpyHostToDevice, cuda_ctx->stream());
        cudaMemcpyAsync(d_perm, perm.data(), ndim * sizeof(int64_t),
                        cudaMemcpyHostToDevice, cuda_ctx->stream());

        const int64_t total = in_shape.numel();
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;

        transpose_kernel<<<blocks, threads, 0, cuda_ctx->stream()>>>(
            static_cast<const float*>(input->data()),
            static_cast<float*>(output->data()),
            d_in_strides,
            d_out_strides,
            d_perm,
            static_cast<int>(ndim),
            total
        );

        // Synchronize and free device memory
        cudaStreamSynchronize(cuda_ctx->stream());
        cudaFree(d_in_strides);
        cudaFree(d_out_strides);
        cudaFree(d_perm);

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(TransposeCUDAPlugin, "Transpose", kTRANSPOSE, CUDA)

}  // namespace operators
}  // namespace mini_infer
