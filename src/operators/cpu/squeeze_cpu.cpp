#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>
#include <cstring>

namespace mini_infer {
namespace operators {

/**
 * @brief Squeeze CPU Plugin
 *
 * Removes dimensions of size 1 from the input tensor.
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
        return 1;  // data only, axes from param
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {

        if (input_shapes.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_shape = input_shapes[0];
        const int64_t ndim = static_cast<int64_t>(input_shape.ndim());

        std::vector<int64_t> output_dims;

        if (param_ && !param_->axes.empty()) {
            // Squeeze specified axes
            std::vector<bool> squeeze_axis(ndim, false);
            for (int64_t axis : param_->axes) {
                if (axis < 0) {
                    axis += ndim;
                }
                if (axis >= 0 && axis < ndim) {
                    squeeze_axis[axis] = true;
                }
            }

            for (int64_t i = 0; i < ndim; ++i) {
                if (squeeze_axis[i]) {
                    if (input_shape[i] != 1) {
                        return core::Status::ERROR_INVALID_ARGUMENT;
                    }
                    // Skip this dimension
                } else {
                    output_dims.push_back(input_shape[i]);
                }
            }
        } else {
            // Squeeze all dimensions of size 1
            for (int64_t i = 0; i < ndim; ++i) {
                if (input_shape[i] != 1) {
                    output_dims.push_back(input_shape[i]);
                }
            }
        }

        // Handle scalar case
        if (output_dims.empty()) {
            output_dims.push_back(1);
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(output_dims));
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

        // Squeeze is just a reshape - copy data
        const size_t size_bytes = input->size_in_bytes();
        std::memcpy(output->data(), input->data(), size_bytes);

        return core::Status::SUCCESS;
    }
};

/**
 * @brief Unsqueeze CPU Plugin
 *
 * Inserts dimensions of size 1 at specified positions.
 */
class UnsqueezeCPUPlugin : public CPUPlugin<UnsqueezeCPUPlugin, SqueezeParam> {
public:
    UnsqueezeCPUPlugin() {
        param_ = std::make_shared<SqueezeParam>();
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
        return 1;  // data only, axes from param
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {

        if (input_shapes.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_shape = input_shapes[0];
        const int64_t in_ndim = static_cast<int64_t>(input_shape.ndim());

        if (!param_ || param_->axes.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const int64_t out_ndim = in_ndim + static_cast<int64_t>(param_->axes.size());

        // Normalize axes
        std::vector<int64_t> normalized_axes;
        for (int64_t axis : param_->axes) {
            if (axis < 0) {
                axis += out_ndim;
            }
            if (axis < 0 || axis >= out_ndim) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
            normalized_axes.push_back(axis);
        }

        // Sort axes
        std::sort(normalized_axes.begin(), normalized_axes.end());

        // Build output shape
        std::vector<int64_t> output_dims;
        output_dims.reserve(out_ndim);

        size_t in_idx = 0;
        size_t axes_idx = 0;

        for (int64_t i = 0; i < out_ndim; ++i) {
            if (axes_idx < normalized_axes.size() && normalized_axes[axes_idx] == i) {
                output_dims.push_back(1);
                axes_idx++;
            } else {
                if (in_idx < input_shape.ndim()) {
                    output_dims.push_back(input_shape[in_idx]);
                    in_idx++;
                }
            }
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(output_dims));
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

        // Unsqueeze is just a reshape - copy data
        const size_t size_bytes = input->size_in_bytes();
        std::memcpy(output->data(), input->data(), size_bytes);

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(SqueezeCPUPlugin, "Squeeze", kSQUEEZE, CPU)
REGISTER_PLUGIN_SIMPLE(UnsqueezeCPUPlugin, "Unsqueeze", kUNSQUEEZE, CPU)

}  // namespace operators
}  // namespace mini_infer
