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

    core::Status infer_output_shapes_with_tensors(
        const std::vector<core::Shape>& input_shapes,
        const std::vector<core::DataType>& input_dtypes,
        const std::vector<std::shared_ptr<core::Tensor>>& input_tensors,
        std::vector<core::Shape>& output_shapes,
        std::vector<core::DataType>& output_dtypes) const override {
        (void)input_dtypes;

        if (input_shapes.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_shape = input_shapes[0];
        int64_t total_elements = input_shape.numel();

        std::vector<int64_t> target_shape;

        // Try to get shape from second input tensor (dynamic shape)
        if (input_tensors.size() >= 2 && input_tensors[1] && input_tensors[1]->data()) {
            const auto& shape_tensor = input_tensors[1];
            size_t num_dims = static_cast<size_t>(shape_tensor->shape().numel());

            // Read shape values based on dtype
            if (shape_tensor->dtype() == core::DataType::INT64) {
                const int64_t* shape_data = static_cast<const int64_t*>(shape_tensor->data());
                target_shape.assign(shape_data, shape_data + num_dims);
            } else if (shape_tensor->dtype() == core::DataType::INT32) {
                const int32_t* shape_data = static_cast<const int32_t*>(shape_tensor->data());
                for (size_t i = 0; i < num_dims; ++i) {
                    target_shape.push_back(static_cast<int64_t>(shape_data[i]));
                }
            } else {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
        } else if (param_ && !param_->shape.empty()) {
            // Fall back to static shape from param
            target_shape = param_->shape;
        } else {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        std::vector<int64_t> resolved_shape;
        auto status = resolve_shape(target_shape, total_elements, resolved_shape);
        if (status != core::Status::SUCCESS) {
            return status;
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(resolved_shape));

        // Output dtype is same as input dtype
        output_dtypes.clear();
        if (!input_dtypes.empty()) {
            output_dtypes.push_back(input_dtypes[0]);
        } else {
            output_dtypes.push_back(core::DataType::FLOAT32);
        }

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
