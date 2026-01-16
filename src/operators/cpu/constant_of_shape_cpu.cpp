#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <cstring>

namespace mini_infer {
namespace operators {

/**
 * @brief ConstantOfShape CPU Plugin
 *
 * Generates a tensor with a given shape filled with a constant value.
 */
class ConstantOfShapeCPUPlugin : public CPUPlugin<ConstantOfShapeCPUPlugin, ConstantOfShapeParam> {
public:
    ConstantOfShapeCPUPlugin() {
        param_ = std::make_shared<ConstantOfShapeParam>();
    }
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
        return 1;  // shape tensor
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {

        if (input_shapes.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Input is a 1D tensor containing the desired output shape
        // We can't infer the exact output shape at compile time if shape is dynamic
        // Return a placeholder shape
        output_shapes.clear();
        output_shapes.push_back(core::Shape({1}));
        return core::Status::SUCCESS;
    }

    core::Status infer_output_metadata(
        const std::vector<core::Shape>& input_shapes,
        const std::vector<core::DataType>& input_dtypes,
        std::vector<core::Shape>& output_shapes,
        std::vector<core::DataType>& output_dtypes) const override {
        (void)input_dtypes;

        auto status = infer_output_shapes(input_shapes, output_shapes);
        if (status != core::Status::SUCCESS) {
            return status;
        }

        output_dtypes.clear();
        output_dtypes.push_back(param_ ? param_->dtype : core::DataType::FLOAT32);
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;

        if (inputs.size() != 1 || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& shape_tensor = inputs[0];
        auto& output = outputs[0];

        if (!shape_tensor || !output || !param_) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Extract target shape from input tensor
        std::vector<int64_t> target_shape;
        const int64_t shape_size = shape_tensor->shape().numel();

        if (shape_tensor->dtype() == core::DataType::INT64) {
            const int64_t* shape_data = static_cast<const int64_t*>(shape_tensor->data());
            for (int64_t i = 0; i < shape_size; ++i) {
                target_shape.push_back(shape_data[i]);
            }
        } else if (shape_tensor->dtype() == core::DataType::INT32) {
            const int32_t* shape_data = static_cast<const int32_t*>(shape_tensor->data());
            for (int64_t i = 0; i < shape_size; ++i) {
                target_shape.push_back(static_cast<int64_t>(shape_data[i]));
            }
        } else {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Resize output if needed
        core::Shape desired_shape(target_shape);
        if (output->shape() != desired_shape) {
            output->resize(desired_shape);
        }

        const int64_t total = output->shape().numel();

        // Fill with constant value
        if (param_->dtype == core::DataType::FLOAT32) {
            float* out_data = static_cast<float*>(output->data());
            std::fill(out_data, out_data + total, param_->value);
        } else if (param_->dtype == core::DataType::INT32) {
            int32_t* out_data = static_cast<int32_t*>(output->data());
            const int32_t int_value = static_cast<int32_t>(param_->value);
            std::fill(out_data, out_data + total, int_value);
        } else if (param_->dtype == core::DataType::INT64) {
            int64_t* out_data = static_cast<int64_t*>(output->data());
            const int64_t int_value = static_cast<int64_t>(param_->value);
            std::fill(out_data, out_data + total, int_value);
        } else {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(ConstantOfShapeCPUPlugin, "ConstantOfShape", kCONSTANT_OF_SHAPE, CPU)

}  // namespace operators
}  // namespace mini_infer
