#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <cmath>

namespace mini_infer {
namespace operators {

namespace {

// GELU approximation using tanh (faster, used by BERT)
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline float gelu_tanh(float x) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

}  // namespace

/**
 * @brief GELU CPU Plugin
 *
 * Implements Gaussian Error Linear Unit activation.
 * Uses the tanh approximation for better performance.
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
class GELUCPUPlugin : public SimpleCPUPlugin<GELUCPUPlugin> {
public:
    GELUCPUPlugin() = default;
    ~GELUCPUPlugin() override = default;

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
            out[i] = gelu_tanh(in[i]);
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(GELUCPUPlugin, "Gelu", kGELU, CPU)

}  // namespace operators
}  // namespace mini_infer
