#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>

namespace mini_infer {
namespace operators {

/**
 * @brief Unsqueeze CPU Plugin
 *
 * Inserts dimensions of size 1 at specified positions.
 */
class UnsqueezeCPUPlugin : public CPUPlugin<UnsqueezeCPUPlugin, UnsqueezeParam> {
public:
    UnsqueezeCPUPlugin() {
        param_ = std::make_shared<UnsqueezeParam>();
    }
    ~UnsqueezeCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Unsqueeze";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kUNSQUEEZE;
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

        if (!param_ || param_->axes.empty()) {
            // No axes specified - return input shape unchanged
            output_shapes.clear();
            output_shapes.push_back(input_shapes[0]);
            return core::Status::SUCCESS;
        }

        const auto& in_dims = input_shapes[0].dims();
        int64_t in_ndim = static_cast<int64_t>(in_dims.size());
        int64_t out_ndim = in_ndim + static_cast<int64_t>(param_->axes.size());

        // Normalize and sort axes
        std::vector<int64_t> axes = param_->axes;
        for (auto& axis : axes) {
            if (axis < 0) axis += out_ndim;
            if (axis < 0 || axis >= out_ndim) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
        }
        std::sort(axes.begin(), axes.end());

        // Build output shape
        std::vector<int64_t> out_dims;
        out_dims.reserve(out_ndim);

        size_t in_idx = 0;
        size_t axes_idx = 0;
        for (int64_t i = 0; i < out_ndim; ++i) {
            if (axes_idx < axes.size() && axes[axes_idx] == i) {
                out_dims.push_back(1);
                ++axes_idx;
            } else {
                if (in_idx < in_dims.size()) {
                    out_dims.push_back(in_dims[in_idx++]);
                }
            }
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(out_dims));
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

        // Unsqueeze is a view operation - just copy data
        const size_t total_bytes = input->shape().numel() * sizeof(float);
        std::memcpy(output->data(), input->data(), total_bytes);

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(UnsqueezeCPUPlugin, "Unsqueeze", kUNSQUEEZE, CPU)

}  // namespace operators
}  // namespace mini_infer
