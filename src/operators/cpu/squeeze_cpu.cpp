#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>

namespace mini_infer {
namespace operators {

/**
 * @brief Squeeze CPU Plugin
 *
 * Removes dimensions of size 1 from the tensor shape.
 */
class SqueezeCPUPlugin : public CPUPlugin<SqueezeCPUPlugin, SqueezeParam> {
public:
    SqueezeCPUPlugin() {
        param_ = std::make_shared<SqueezeParam>();
    }
    ~SqueezeCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Squeeze";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kSQUEEZE;
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

        const auto& in_dims = input_shapes[0].dims();
        int64_t ndim = static_cast<int64_t>(in_dims.size());

        std::vector<int64_t> out_dims;

        if (param_ && !param_->axes.empty()) {
            // Squeeze specific axes
            std::vector<bool> squeeze_axis(ndim, false);
            for (int64_t axis : param_->axes) {
                if (axis < 0) axis += ndim;
                if (axis >= 0 && axis < ndim && in_dims[axis] == 1) {
                    squeeze_axis[axis] = true;
                }
            }
            for (int64_t i = 0; i < ndim; ++i) {
                if (!squeeze_axis[i]) {
                    out_dims.push_back(in_dims[i]);
                }
            }
        } else {
            // Squeeze all size-1 dimensions
            for (int64_t i = 0; i < ndim; ++i) {
                if (in_dims[i] != 1) {
                    out_dims.push_back(in_dims[i]);
                }
            }
        }

        // Ensure at least 1D output
        if (out_dims.empty()) {
            out_dims.push_back(1);
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

        // Squeeze is a view operation - just copy data
        const size_t total_bytes = input->shape().numel() * sizeof(float);
        std::memcpy(output->data(), input->data(), total_bytes);

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(SqueezeCPUPlugin, "Squeeze", kSQUEEZE, CPU)

}  // namespace operators
}  // namespace mini_infer
