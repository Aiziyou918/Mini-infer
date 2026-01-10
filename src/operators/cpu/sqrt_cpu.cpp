#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <cmath>

namespace mini_infer {
namespace operators {

/**
 * @brief Sqrt CPU Plugin
 *
 * Implements element-wise square root.
 * output = sqrt(input)
 */
class SqrtCPUPlugin : public SimpleCPUPlugin<SqrtCPUPlugin> {
public:
    SqrtCPUPlugin() = default;
    ~SqrtCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Sqrt";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kSQRT;
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

        output_shapes.clear();
        output_shapes.push_back(input_shapes[0]);
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

        const auto& input = inputs[0];
        auto& output = outputs[0];

        if (!input || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        if (input->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& shape = input->shape();
        const size_t total = static_cast<size_t>(shape.numel());

        const float* in = static_cast<const float*>(input->data());
        float* out = static_cast<float*>(output->data());

        for (size_t i = 0; i < total; ++i) {
            out[i] = std::sqrt(in[i]);
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(SqrtCPUPlugin, "Sqrt", kSQRT, CPU)

}  // namespace operators
}  // namespace mini_infer
