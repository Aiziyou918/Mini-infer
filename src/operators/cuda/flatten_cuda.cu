#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace operators {

/**
 * @brief Flatten CUDA Plugin
 *
 * Flattens the input tensor into a 2D matrix.
 * Uses cudaMemcpyAsync for efficient device-to-device copy.
 */
class FlattenCUDAPlugin : public CUDAPlugin<FlattenCUDAPlugin, FlattenParam> {
public:
    FlattenCUDAPlugin() {
        param_ = std::make_shared<FlattenParam>();
    }
    ~FlattenCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Flatten";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kFLATTEN;
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

        const auto& input_shape = input_shapes[0];
        const int ndim = static_cast<int>(input_shape.ndim());

        int axis = param_ ? param_->axis : 1;
        if (axis < 0) {
            axis += ndim;
        }
        if (axis < 0 || axis > ndim) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        int64_t outer_dim = 1;
        int64_t inner_dim = 1;

        for (int i = 0; i < axis; ++i) {
            outer_dim *= input_shape[i];
        }
        for (int i = axis; i < ndim; ++i) {
            inner_dim *= input_shape[i];
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape({outer_dim, inner_dim}));
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

        if (!input || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        if (!context.device_context) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(
            context.device_context
        );
        if (!cuda_ctx) {
            return core::Status::ERROR_BACKEND;
        }

        const void* src = input->data();
        void* dst = output->data();
        size_t size_bytes = input->size_in_bytes();

        // Skip if same memory location
        if (src == dst || !src || !dst) {
            return core::Status::SUCCESS;
        }

        // Use cudaMemcpyAsync for best performance
        cudaMemcpyAsync(dst, src, size_bytes, cudaMemcpyDeviceToDevice, cuda_ctx->stream());

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] Flatten plugin error: " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }
};

// Define creator and register plugin
REGISTER_PLUGIN_SIMPLE(FlattenCUDAPlugin, "Flatten", kFLATTEN, CUDA)

}  // namespace operators
}  // namespace mini_infer
