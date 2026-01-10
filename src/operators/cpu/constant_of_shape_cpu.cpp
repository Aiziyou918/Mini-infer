#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

namespace mini_infer {
namespace operators {

/**
 * @brief ConstantOfShape CPU Plugin
 *
 * Creates a tensor of the given shape filled with a constant value.
 * Input: shape tensor (1D INT64)
 * Output: tensor of the specified shape filled with constant value (default 0)
 */
class ConstantOfShapeCPUPlugin : public CPUPlugin<ConstantOfShapeCPUPlugin, PluginParam> {
public:
    ConstantOfShapeCPUPlugin() = default;
    ~ConstantOfShapeCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "ConstantOfShape";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kCONSTANT_OF_SHAPE;
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

        // Input is a 1D shape tensor, output shape is determined at runtime
        // For shape inference, we return a dynamic shape based on input
        // The actual shape will be determined when the shape tensor data is available
        output_shapes.clear();

        // If input shape is known (e.g., [3]), output will have 3 dimensions
        // But we don't know the actual dimension values until runtime
        // Return a placeholder shape with the correct number of dimensions
        int64_t ndim = input_shapes[0].numel();
        if (ndim > 0) {
            std::vector<int64_t> dims(static_cast<size_t>(ndim), -1);  // Dynamic dimensions
            output_shapes.push_back(core::Shape(dims));
        } else {
            // Scalar output
            output_shapes.push_back(core::Shape({}));
        }

        return core::Status::SUCCESS;
    }

    core::Status infer_output_metadata(
        const std::vector<core::Shape>& input_shapes,
        const std::vector<core::DataType>& /*input_dtypes*/,
        std::vector<core::Shape>& output_shapes,
        std::vector<core::DataType>& output_dtypes) const override {
        auto status = infer_output_shapes(input_shapes, output_shapes);
        if (status != core::Status::SUCCESS) {
            return status;
        }
        // Default output type is FLOAT32
        output_dtypes.clear();
        output_dtypes.push_back(core::DataType::FLOAT32);
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

        const auto& shape_tensor = inputs[0];
        auto& output = outputs[0];

        if (!shape_tensor || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Fill output with zeros (default constant value)
        float* out_ptr = static_cast<float*>(output->data());
        int64_t total = output->shape().numel();

        for (int64_t i = 0; i < total; ++i) {
            out_ptr[i] = 0.0f;
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(ConstantOfShapeCPUPlugin, "ConstantOfShape", kCONSTANT_OF_SHAPE, CPU)

}  // namespace operators
}  // namespace mini_infer
