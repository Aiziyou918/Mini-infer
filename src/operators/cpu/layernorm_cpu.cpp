#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <cmath>
#include <numeric>

namespace mini_infer {
namespace operators {

/**
 * @brief LayerNorm CPU Plugin
 *
 * Performs Layer Normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
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
        return 1;  // Only Y output, ignore Mean and InvStdDev
    }

    int32_t get_nb_inputs() const noexcept override {
        return 3;  // X, Scale, Bias
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {

        if (input_shapes.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Output shape is same as input shape
        output_shapes.clear();
        output_shapes.push_back(input_shapes[0]);
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;

        if (inputs.size() < 2 || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input = inputs[0];
        const auto& scale = inputs[1];
        const auto* bias = (inputs.size() > 2) ? inputs[2].get() : nullptr;
        auto& output = outputs[0];

        if (!input || !scale || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        if (input->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const core::Shape& in_shape = input->shape();
        const int64_t ndim = static_cast<int64_t>(in_shape.ndim());

        // Get axis (default: -1, normalize over last axis)
        int64_t axis = param_ ? param_->axis : -1;
        if (axis < 0) {
            axis += ndim;
        }
        if (axis < 0 || axis >= ndim) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        float epsilon = param_ ? param_->epsilon : 1e-5f;

        // Compute sizes
        int64_t outer_size = 1;
        for (int64_t i = 0; i < axis; ++i) {
            outer_size *= in_shape[i];
        }

        int64_t norm_size = 1;
        for (int64_t i = axis; i < ndim; ++i) {
            norm_size *= in_shape[i];
        }

        const float* in_data = static_cast<const float*>(input->data());
        const float* scale_data = static_cast<const float*>(scale->data());
        const float* bias_data = bias ? static_cast<const float*>(bias->data()) : nullptr;
        float* out_data = static_cast<float*>(output->data());

        // Process each outer slice
        for (int64_t o = 0; o < outer_size; ++o) {
            const float* slice_in = in_data + o * norm_size;
            float* slice_out = out_data + o * norm_size;

            // Compute mean
            float mean = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                mean += slice_in[i];
            }
            mean /= static_cast<float>(norm_size);

            // Compute variance
            float var = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                float diff = slice_in[i] - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(norm_size);

            // Compute inverse standard deviation
            float inv_std = 1.0f / std::sqrt(var + epsilon);

            // Normalize and apply scale/bias
            for (int64_t i = 0; i < norm_size; ++i) {
                float normalized = (slice_in[i] - mean) * inv_std;
                float scaled = normalized * scale_data[i];
                slice_out[i] = bias_data ? (scaled + bias_data[i]) : scaled;
            }
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(LayerNormCPUPlugin, "LayerNormalization", kLAYER_NORM, CPU)

}  // namespace operators
}  // namespace mini_infer
