#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace mini_infer {
namespace operators {

/**
 * @brief ReduceMean CPU Plugin
 *
 * Computes the mean of elements along specified axes.
 */
class ReduceMeanCPUPlugin : public CPUPlugin<ReduceMeanCPUPlugin, ReduceMeanParam> {
public:
    ReduceMeanCPUPlugin() {
        param_ = std::make_shared<ReduceMeanParam>();
    }
    ~ReduceMeanCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "ReduceMean";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kREDUCE_MEAN;
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

        const auto& input_shape = input_shapes[0];
        const int64_t ndim = static_cast<int64_t>(input_shape.ndim());

        if (!param_) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Normalize axes
        std::vector<int64_t> axes = param_->axes;
        if (axes.empty()) {
            // Reduce all dimensions
            for (int64_t i = 0; i < ndim; ++i) {
                axes.push_back(i);
            }
        }

        std::vector<bool> reduce_axis(ndim, false);
        for (int64_t axis : axes) {
            if (axis < 0) {
                axis += ndim;
            }
            if (axis >= 0 && axis < ndim) {
                reduce_axis[axis] = true;
            }
        }

        std::vector<int64_t> output_dims;
        for (int64_t i = 0; i < ndim; ++i) {
            if (reduce_axis[i]) {
                if (param_->keepdims) {
                    output_dims.push_back(1);
                }
            } else {
                output_dims.push_back(input_shape[i]);
            }
        }

        // Handle scalar output
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

        if (!input || !output || !param_) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        if (input->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const core::Shape& in_shape = input->shape();
        const int64_t ndim = static_cast<int64_t>(in_shape.ndim());

        // Normalize axes
        std::vector<int64_t> axes = param_->axes;
        if (axes.empty()) {
            for (int64_t i = 0; i < ndim; ++i) {
                axes.push_back(i);
            }
        }

        std::vector<bool> reduce_axis(ndim, false);
        for (int64_t axis : axes) {
            if (axis < 0) axis += ndim;
            if (axis >= 0 && axis < ndim) {
                reduce_axis[axis] = true;
            }
        }

        // Compute reduction count
        int64_t reduce_count = 1;
        for (int64_t i = 0; i < ndim; ++i) {
            if (reduce_axis[i]) {
                reduce_count *= in_shape[i];
            }
        }

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        const core::Shape& out_shape = output->shape();
        const int64_t out_total = out_shape.numel();

        // Initialize output to zero
        std::fill(out_data, out_data + out_total, 0.0f);

        // Compute input strides
        std::vector<int64_t> in_strides(ndim);
        int64_t stride = 1;
        for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
            in_strides[i] = stride;
            stride *= in_shape[i];
        }

        // Iterate over all input elements
        const int64_t in_total = in_shape.numel();
        std::vector<int64_t> in_indices(ndim, 0);

        for (int64_t i = 0; i < in_total; ++i) {
            // Compute output index (skip reduced dimensions)
            int64_t out_idx = 0;
            int64_t out_stride = 1;
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                if (!reduce_axis[d]) {
                    // Find position in output
                    int64_t out_pos = 0;
                    int64_t out_dim_stride = 1;
                    for (int od = static_cast<int>(out_shape.ndim()) - 1; od >= 0; --od) {
                        // Map input dimension to output dimension
                        int64_t in_dim_idx = d;
                        int64_t out_dim_idx = od;

                        // Count non-reduced dimensions up to d
                        int64_t non_reduced_count = 0;
                        for (int64_t dd = 0; dd <= d; ++dd) {
                            if (!reduce_axis[dd]) {
                                non_reduced_count++;
                            }
                        }

                        if (param_->keepdims) {
                            out_idx += in_indices[d] * out_stride;
                        }
                    }
                }
            }

            // Simpler approach: compute output index directly
            out_idx = 0;
            int64_t out_mult = 1;
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                if (param_->keepdims) {
                    if (!reduce_axis[d]) {
                        out_idx += in_indices[d] * out_mult;
                    }
                    out_mult *= (reduce_axis[d] ? 1 : in_shape[d]);
                } else {
                    if (!reduce_axis[d]) {
                        out_idx += in_indices[d] * out_mult;
                        out_mult *= in_shape[d];
                    }
                }
            }

            out_data[out_idx] += in_data[i];

            // Increment input indices
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                in_indices[d]++;
                if (in_indices[d] < in_shape[d]) {
                    break;
                }
                in_indices[d] = 0;
            }
        }

        // Divide by count to get mean
        for (int64_t i = 0; i < out_total; ++i) {
            out_data[i] /= static_cast<float>(reduce_count);
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(ReduceMeanCPUPlugin, "ReduceMean", kREDUCE_MEAN, CPU)

}  // namespace operators
}  // namespace mini_infer
