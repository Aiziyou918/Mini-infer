#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>

namespace mini_infer {
namespace operators {

/**
 * @brief Expand CPU Plugin
 *
 * Broadcasts input tensor to a specified shape.
 * Input 0: data tensor
 * Input 1: shape tensor (1D INT64)
 */
class ExpandCPUPlugin : public SimpleCPUPlugin<ExpandCPUPlugin> {
public:
    ExpandCPUPlugin() = default;
    ~ExpandCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Expand";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kEXPAND;
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

        // Input 0 is the data tensor
        // Input 1 is the shape tensor (1D)
        // Output shape is determined by broadcasting input[0] with the target shape

        const auto& data_shape = input_shapes[0];
        const auto& shape_tensor_shape = input_shapes[1];

        // The shape tensor tells us the target number of dimensions
        int64_t target_ndim = shape_tensor_shape.numel();

        if (target_ndim <= 0) {
            // If shape tensor is empty or scalar, use input shape
            output_shapes.clear();
            output_shapes.push_back(data_shape);
            return core::Status::SUCCESS;
        }

        // Create output shape with dynamic dimensions
        // Actual values will be determined at runtime from shape tensor data
        size_t ndim_out = std::max(data_shape.ndim(), static_cast<size_t>(target_ndim));
        std::vector<int64_t> out_dims(ndim_out, -1);  // Dynamic dimensions

        output_shapes.clear();
        output_shapes.push_back(core::Shape(out_dims));
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;

        if (inputs.size() != 2 || outputs.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& data = inputs[0];
        auto& output = outputs[0];

        if (!data || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Simplified implementation: broadcast copy
        if (data->dtype() == core::DataType::FLOAT32) {
            const float* data_ptr = static_cast<const float*>(data->data());
            float* out_ptr = static_cast<float*>(output->data());
            int64_t data_total = data->shape().numel();
            int64_t out_total = output->shape().numel();

            for (int64_t i = 0; i < out_total; ++i) {
                out_ptr[i] = data_ptr[i % data_total];
            }
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(ExpandCPUPlugin, "Expand", kEXPAND, CPU)

}  // namespace operators
}  // namespace mini_infer
