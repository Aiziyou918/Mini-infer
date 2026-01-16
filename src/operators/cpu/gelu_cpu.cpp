#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <cmath>

namespace mini_infer {
namespace operators {

/**
 * @brief Gelu CPU Plugin
 *
 * Implements the Gaussian Error Linear Unit (GELU) activation function.
 * GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal distribution.
 *
 * Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 */
class GeluCPUPlugin : public SimpleCPUPlugin<GeluCPUPlugin> {
public:
    GeluCPUPlugin() = default;
    ~GeluCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Gelu";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kGELU;
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

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        const int64_t total = input->shape().numel();

        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/pi)
        constexpr float coeff = 0.044715f;

        for (int64_t i = 0; i < total; ++i) {
            const float x = in_data[i];
            const float x_cubed = x * x * x;
            const float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            out_data[i] = 0.5f * x * (1.0f + std::tanh(inner));
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(GeluCPUPlugin, "Gelu", kGELU, CPU)

}  // namespace operators
}  // namespace mini_infer
