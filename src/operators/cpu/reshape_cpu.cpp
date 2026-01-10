#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <cstring>

namespace mini_infer {
namespace operators {

/**
 * @brief Reshape CPU Plugin
 *
 * Reshapes input tensor to target shape without copying data (when possible).
 * Supports -1 in shape to infer dimension size.
 */
class ReshapeCPUPlugin : public CPUPlugin<ReshapeCPUPlugin, ReshapeParam> {
public:
    ReshapeCPUPlugin() {
        param_ = std::make_shared<ReshapeParam>();
    }
    ~ReshapeCPUPlugin() override = default;

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
        return 1;  // Can be 2 if shape is provided as tensor
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
            // Dynamic reshape - shape comes from input tensor at runtime
            // Check if we have a second input (shape tensor)
            if (input_shapes.size() >= 2) {
                // The shape tensor tells us the output ndim
                const auto& shape_tensor_shape = input_shapes[1];
                if (shape_tensor_shape.ndim() == 1 && shape_tensor_shape.dims()[0] > 0) {
                    // Output has shape_tensor_shape[0] dimensions, all dynamic
                    int64_t out_ndim = shape_tensor_shape.dims()[0];
                    std::vector<int64_t> dynamic_shape(out_ndim, -1);
                    output_shapes.clear();
                    output_shapes.push_back(core::Shape(dynamic_shape));
                    return core::Status::SUCCESS;
                }
            }
            // Fallback: use input ndim as a guess
            output_shapes.clear();
            std::vector<int64_t> dynamic_shape(input_shape.ndim(), -1);
            output_shapes.push_back(core::Shape(dynamic_shape));
            return core::Status::SUCCESS;
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
        (void)context;

        if (inputs.empty() || outputs.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input = inputs[0];
        auto& output = outputs[0];

        if (!input || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const void* src = input->data();
        void* dst = output->data();

        if (src && dst && src != dst) {
            std::memcpy(dst, src, input->size_in_bytes());
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
                // Keep original dimension (not supported in this simple version)
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

        // Verify total elements match
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
REGISTER_PLUGIN_SIMPLE(ReshapeCPUPlugin, "Reshape", kRESHAPE, CPU)

}  // namespace operators
}  // namespace mini_infer
