#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <algorithm>
#include <string>

namespace mini_infer {
namespace operators {

namespace {

__global__ void copy_kernel(const float* input, float* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

}  // namespace

/**
 * @brief Unsqueeze CUDA Plugin
 */
class UnsqueezeCUDAPlugin : public CUDAPlugin<UnsqueezeCUDAPlugin, UnsqueezeParam> {
public:
    UnsqueezeCUDAPlugin() {
        param_ = std::make_shared<UnsqueezeParam>();
    }
    ~UnsqueezeCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Unsqueeze";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kUNSQUEEZE;
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

        if (!param_ || param_->axes.empty()) {
            output_shapes.clear();
            output_shapes.push_back(input_shapes[0]);
            return core::Status::SUCCESS;
        }

        const auto& in_dims = input_shapes[0].dims();
        int64_t in_ndim = static_cast<int64_t>(in_dims.size());
        int64_t out_ndim = in_ndim + static_cast<int64_t>(param_->axes.size());

        std::vector<int64_t> axes = param_->axes;
        for (auto& axis : axes) {
            if (axis < 0) axis += out_ndim;
            if (axis < 0 || axis >= out_ndim) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
        }
        std::sort(axes.begin(), axes.end());

        std::vector<int64_t> out_dims;
        out_dims.reserve(out_ndim);

        size_t in_idx = 0;
        size_t axes_idx = 0;
        for (int64_t i = 0; i < out_ndim; ++i) {
            if (axes_idx < axes.size() && axes[axes_idx] == i) {
                out_dims.push_back(1);
                ++axes_idx;
            } else {
                if (in_idx < in_dims.size()) {
                    out_dims.push_back(in_dims[in_idx++]);
                }
            }
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

        const int64_t n = input->shape().numel();
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;

        copy_kernel<<<blocks, threads, 0, cuda_ctx->stream()>>>(
            static_cast<const float*>(input->data()),
            static_cast<float*>(output->data()),
            n
        );

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(UnsqueezeCUDAPlugin, "Unsqueeze", kUNSQUEEZE, CUDA)

}  // namespace operators
}  // namespace mini_infer
