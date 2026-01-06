#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace operators {

/**
 * @brief Reshape CUDA Plugin
 *
 * Reshapes input tensor to target shape.
 * Uses cudaMemcpyAsync for efficient device-to-device copy.
 */
class ReshapeCUDAPlugin : public CUDAPlugin<ReshapeCUDAPlugin, ReshapeParam> {
public:
    ReshapeCUDAPlugin() {
        param_ = std::make_shared<ReshapeParam>();
    }
    ~ReshapeCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Reshape";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kRESHAPE;
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
        if (input_shapes.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_shape = input_shapes[0];
        int64_t total_elements = input_shape.numel();

        if (!param_ || param_->shape.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        std::vector<int64_t> resolved_shape;
        auto status = resolve_shape(param_->shape, total_elements, resolved_shape);
        if (status != core::Status::SUCCESS) {
            return status;
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(resolved_shape));
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {

        if (inputs.empty() || outputs.empty()) {
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
            MI_LOG_ERROR("[CUDA] Reshape plugin error: " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }

private:
    core::Status resolve_shape(
        const std::vector<int64_t>& target_shape,
        int64_t total_elements,
        std::vector<int64_t>& resolved_shape) const {

        resolved_shape = target_shape;
        int64_t known_product = 1;
        int infer_idx = -1;

        for (size_t i = 0; i < resolved_shape.size(); ++i) {
            if (resolved_shape[i] == -1) {
                if (infer_idx != -1) {
                    return core::Status::ERROR_INVALID_ARGUMENT;
                }
                infer_idx = static_cast<int>(i);
            } else if (resolved_shape[i] == 0) {
                return core::Status::ERROR_NOT_IMPLEMENTED;
            } else {
                known_product *= resolved_shape[i];
            }
        }

        if (infer_idx != -1) {
            if (known_product == 0 || total_elements % known_product != 0) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
            resolved_shape[infer_idx] = total_elements / known_product;
        }

        int64_t output_elements = 1;
        for (auto dim : resolved_shape) {
            output_elements *= dim;
        }
        if (output_elements != total_elements) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        return core::Status::SUCCESS;
    }
};

// Define creator and register plugin
REGISTER_PLUGIN_SIMPLE(ReshapeCUDAPlugin, "Reshape", kRESHAPE, CUDA)

}  // namespace operators
}  // namespace mini_infer
