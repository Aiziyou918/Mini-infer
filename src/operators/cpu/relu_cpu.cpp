#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>

namespace mini_infer {
namespace operators {

/**
 * @brief ReLU CPU Plugin
 *
 * Implements ReLU activation: output = max(0, input)
 * Supports FLOAT32 and INT32 data types.
 */
class ReLUCPUPlugin : public SimpleCPUPlugin<ReLUCPUPlugin> {
public:
    ReLUCPUPlugin() = default;
    ~ReLUCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Relu";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kRELU;
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

        const auto& shape = input->shape();
        const size_t total = static_cast<size_t>(shape.numel());

        switch (input->dtype()) {
            case core::DataType::FLOAT32: {
                const float* in = static_cast<const float*>(input->data());
                float* out = static_cast<float*>(output->data());
                for (size_t i = 0; i < total; ++i) {
                    out[i] = std::max(0.0f, in[i]);
                }
                break;
            }
            case core::DataType::INT32: {
                const int32_t* in = static_cast<const int32_t*>(input->data());
                int32_t* out = static_cast<int32_t*>(output->data());
                for (size_t i = 0; i < total; ++i) {
                    out[i] = std::max(0, in[i]);
                }
                break;
            }
            default:
                return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        return core::Status::SUCCESS;
    }
};

// Define creator and register plugin
REGISTER_PLUGIN_SIMPLE(ReLUCPUPlugin, "Relu", kRELU, CPU)

}  // namespace operators
}  // namespace mini_infer
