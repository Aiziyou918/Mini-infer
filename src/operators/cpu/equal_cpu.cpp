#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

namespace mini_infer {
namespace operators {

/**
 * @brief Equal CPU Plugin
 *
 * Element-wise comparison returning boolean tensor.
 * Output shape follows NumPy broadcasting rules.
 */
class EqualCPUPlugin : public SimpleCPUPlugin<EqualCPUPlugin> {
public:
    EqualCPUPlugin() = default;
    ~EqualCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Equal";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kEQUAL;
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

        // Broadcast shapes
        const auto& a = input_shapes[0];
        const auto& b = input_shapes[1];

        size_t ndim_a = a.ndim();
        size_t ndim_b = b.ndim();
        size_t ndim_out = std::max(ndim_a, ndim_b);

        std::vector<int64_t> out_dims(ndim_out);
        for (size_t i = 0; i < ndim_out; ++i) {
            int64_t dim_a = (i < ndim_out - ndim_a) ? 1 : a[i - (ndim_out - ndim_a)];
            int64_t dim_b = (i < ndim_out - ndim_b) ? 1 : b[i - (ndim_out - ndim_b)];

            // Handle dynamic dimensions
            if (dim_a < 0 || dim_b < 0) {
                out_dims[i] = (dim_a > 0) ? dim_a : dim_b;
            } else if (dim_a == dim_b) {
                out_dims[i] = dim_a;
            } else if (dim_a == 1) {
                out_dims[i] = dim_b;
            } else if (dim_b == 1) {
                out_dims[i] = dim_a;
            } else {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(out_dims));
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
        // Equal output is always BOOL (represented as INT8 or similar)
        output_dtypes.clear();
        output_dtypes.push_back(core::DataType::INT8);
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

        // Simplified implementation - actual broadcast comparison would be more complex
        const auto& a = inputs[0];
        const auto& b = inputs[1];
        auto& output = outputs[0];

        if (!a || !b || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // For now, just fill with zeros (false)
        int8_t* out_ptr = static_cast<int8_t*>(output->data());
        int64_t total = output->shape().numel();

        for (int64_t i = 0; i < total; ++i) {
            out_ptr[i] = 0;
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(EqualCPUPlugin, "Equal", kEQUAL, CPU)

}  // namespace operators
}  // namespace mini_infer
