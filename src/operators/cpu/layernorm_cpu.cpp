#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <cmath>
#include <numeric>

namespace mini_infer {
namespace operators {

/**
 * @brief LayerNorm CPU Plugin
 *
 * Implements Layer Normalization.
 * y = gamma * (x - mean) / sqrt(var + epsilon) + beta
 */
class LayerNormCPUPlugin : public CPUPlugin<LayerNormCPUPlugin, LayerNormParam> {
public:
    LayerNormCPUPlugin() {
        param_ = std::make_shared<LayerNormParam>();
    }
    ~LayerNormCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "LayerNormalization";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kLAYER_NORM;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return 3;  // input, gamma (scale), beta (bias)
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

        if (inputs.size() < 1 || outputs.size() != 1) {
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
        const auto& dims = shape.dims();
        const size_t ndim = dims.size();

        // Get axis (normalize from this axis to the end)
        int64_t axis = param_ ? param_->axis : -1;
        if (axis < 0) axis += static_cast<int64_t>(ndim);
        if (axis < 0 || axis >= static_cast<int64_t>(ndim)) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        float epsilon = param_ ? param_->epsilon : 1e-5f;

        // Compute outer and inner dimensions
        int64_t outer_size = 1;
        int64_t inner_size = 1;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
            outer_size *= dims[i];
        }
        for (size_t i = static_cast<size_t>(axis); i < ndim; ++i) {
            inner_size *= dims[i];
        }

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        // Get gamma and beta if provided
        const float* gamma = nullptr;
        const float* beta = nullptr;
        if (inputs.size() > 1 && inputs[1]) {
            gamma = static_cast<const float*>(inputs[1]->data());
        }
        if (inputs.size() > 2 && inputs[2]) {
            beta = static_cast<const float*>(inputs[2]->data());
        }

        // Process each outer slice
        for (int64_t o = 0; o < outer_size; ++o) {
            const float* slice_in = in_data + o * inner_size;
            float* slice_out = out_data + o * inner_size;

            // Compute mean
            float mean = 0.0f;
            for (int64_t i = 0; i < inner_size; ++i) {
                mean += slice_in[i];
            }
            mean /= static_cast<float>(inner_size);

            // Compute variance
            float var = 0.0f;
            for (int64_t i = 0; i < inner_size; ++i) {
                float diff = slice_in[i] - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(inner_size);

            // Normalize
            float inv_std = 1.0f / std::sqrt(var + epsilon);
            for (int64_t i = 0; i < inner_size; ++i) {
                float normalized = (slice_in[i] - mean) * inv_std;
                if (gamma) normalized *= gamma[i];
                if (beta) normalized += beta[i];
                slice_out[i] = normalized;
            }
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(LayerNormCPUPlugin, "LayerNormalization", kLAYER_NORM, CPU)

}  // namespace operators
}  // namespace mini_infer
