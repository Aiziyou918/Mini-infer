#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>
#include <cmath>

namespace mini_infer {
namespace operators {

/**
 * @brief Softmax CPU Plugin
 *
 * Computes softmax along the specified axis using the max-subtraction trick
 * for numerical stability.
 */
class SoftmaxCPUPlugin : public CPUPlugin<SoftmaxCPUPlugin, SoftmaxParam> {
public:
    SoftmaxCPUPlugin() {
        param_ = std::make_shared<SoftmaxParam>();
    }
    ~SoftmaxCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Softmax";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kSOFTMAX;
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
        if (shape.ndim() == 0) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto dims = shape.dims();
        int axis = param_ ? param_->axis : -1;
        if (axis < 0) {
            axis += static_cast<int>(dims.size());
        }
        if (axis < 0 || axis >= static_cast<int>(dims.size())) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        size_t outer = 1;
        for (int i = 0; i < axis; ++i) {
            outer *= static_cast<size_t>(dims[static_cast<size_t>(i)]);
        }
        size_t inner = 1;
        for (size_t i = static_cast<size_t>(axis + 1); i < dims.size(); ++i) {
            inner *= static_cast<size_t>(dims[i]);
        }
        const size_t dim = static_cast<size_t>(dims[static_cast<size_t>(axis)]);

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());
        if (!in_data || !out_data) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const size_t stride = inner;
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                const size_t base = o * dim * stride + i;

                // Find max for numerical stability
                float max_val = in_data[base];
                for (size_t d = 1; d < dim; ++d) {
                    max_val = std::max(max_val, in_data[base + d * stride]);
                }

                // Compute exp and sum
                float sum = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    float v = std::exp(in_data[base + d * stride] - max_val);
                    out_data[base + d * stride] = v;
                    sum += v;
                }

                // Normalize
                if (sum != 0.0f) {
                    const float inv_sum = 1.0f / sum;
                    for (size_t d = 0; d < dim; ++d) {
                        out_data[base + d * stride] *= inv_sum;
                    }
                }
            }
        }

        return core::Status::SUCCESS;
    }
};

// Define creator and register plugin
REGISTER_PLUGIN_SIMPLE(SoftmaxCPUPlugin, "Softmax", kSOFTMAX, CPU)

}  // namespace operators
}  // namespace mini_infer
