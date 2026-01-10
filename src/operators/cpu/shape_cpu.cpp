#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

namespace mini_infer {
namespace operators {

/**
 * @brief Shape CPU Plugin
 *
 * Returns the shape of the input tensor as a 1D INT64 tensor.
 * Output shape: [ndim] where ndim is the number of dimensions of the input.
 */
class ShapeCPUPlugin : public CPUPlugin<ShapeCPUPlugin, PluginParam> {
public:
    ShapeCPUPlugin() = default;
    ~ShapeCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Shape";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kSHAPE;
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

        // Output is a 1D tensor with size = number of dimensions of input
        int64_t ndim = static_cast<int64_t>(input_shapes[0].ndim());
        output_shapes.clear();
        output_shapes.push_back(core::Shape({ndim}));
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
        // Shape output is always INT64
        output_dtypes.clear();
        output_dtypes.push_back(core::DataType::INT64);
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

        const auto& input_dims = input->shape().dims();
        int64_t* out_ptr = static_cast<int64_t*>(output->data());

        for (size_t i = 0; i < input_dims.size(); ++i) {
            out_ptr[i] = input_dims[i];
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(ShapeCPUPlugin, "Shape", kSHAPE, CPU)

}  // namespace operators
}  // namespace mini_infer
