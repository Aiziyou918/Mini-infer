#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace mini_infer {
namespace operators {

class SliceCPUPlugin : public SimpleCPUPlugin<SliceCPUPlugin> {
public:
    SliceCPUPlugin() = default;
    ~SliceCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Slice";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kSLICE;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        // Dynamic: 1 (static params) or up to 5 (data, starts, ends, axes, steps)
        return -1;
    }

    void set_param(std::shared_ptr<PluginParam> param) override {
        param_ = std::dynamic_pointer_cast<SliceParam>(param);
    }

    const void* get_param_ptr() const noexcept override {
        return param_.get();
    }

    // Check if we have static parameters
    bool has_static_params() const {
        return param_ && !param_->starts.empty() && !param_->ends.empty();
    }

    // Helper to read int64 data from tensor
    static std::vector<int64_t> read_int64_tensor(const std::shared_ptr<core::Tensor>& tensor) {
        std::vector<int64_t> result;
        if (!tensor || !tensor->data()) return result;

        int64_t numel = tensor->shape().numel();
        const int64_t* data = static_cast<const int64_t*>(tensor->data());
        result.assign(data, data + numel);
        return result;
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {
        if (input_shapes.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_dims = input_shapes[0].dims();
        int64_t ndim = static_cast<int64_t>(input_dims.size());

        // Check if we have static params
        if (!has_static_params()) {
            // Dynamic slice - output shape depends on runtime values
            // Mark sliced dimensions as dynamic (-1)
            std::vector<int64_t> out_dims = input_dims;

            // If we have axes info from param, mark those as dynamic
            if (param_ && !param_->axes.empty()) {
                for (int64_t ax : param_->axes) {
                    if (ax < 0) ax += ndim;
                    if (ax >= 0 && ax < ndim) {
                        out_dims[ax] = -1;
                    }
                }
            } else {
                // No axes info - assume all dimensions could be sliced
                // For safety, mark all as dynamic
                for (auto& d : out_dims) {
                    d = -1;
                }
            }

            output_shapes.clear();
            output_shapes.push_back(core::Shape(out_dims));
            return core::Status::SUCCESS;
        }

        // Static params path
        std::vector<int64_t> axes = param_->axes;
        if (axes.empty()) {
            for (int64_t i = 0; i < static_cast<int64_t>(param_->starts.size()); ++i) {
                axes.push_back(i);
            }
        }
        for (auto& ax : axes) {
            if (ax < 0) ax += ndim;
        }

        std::vector<int64_t> steps = param_->steps;
        if (steps.empty()) {
            steps.resize(axes.size(), 1);
        }

        std::vector<int64_t> out_dims = input_dims;

        for (size_t i = 0; i < axes.size(); ++i) {
            int64_t axis = axes[i];
            if (axis < 0 || axis >= ndim) continue;

            if (i >= param_->starts.size() || i >= param_->ends.size()) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }

            int64_t dim_size = input_dims[axis];
            int64_t start = param_->starts[i];
            int64_t end = param_->ends[i];
            int64_t step = steps[i];

            if (start < 0) start += dim_size;
            start = std::max(int64_t(0), std::min(start, dim_size));

            if (end < 0) end += dim_size;
            if (end > dim_size) end = dim_size;
            end = std::max(int64_t(0), std::min(end, dim_size));

            int64_t out_dim = 0;
            if (step > 0) {
                out_dim = (end - start + step - 1) / step;
            } else if (step < 0) {
                out_dim = (start - end - step - 1) / (-step);
            }
            out_dim = std::max(int64_t(0), out_dim);

            out_dims[axis] = out_dim;
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

        if (inputs.empty() || outputs.empty()) {
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

        const auto& input_shape = input->shape();
        const auto& input_dims = input_shape.dims();
        const auto& output_shape = output->shape();
        const auto& output_dims = output_shape.dims();
        int64_t ndim = static_cast<int64_t>(input_dims.size());

        // Get slice parameters - either from static param or from input tensors
        std::vector<int64_t> starts_vec, ends_vec, axes_vec, steps_vec;

        if (has_static_params()) {
            // Use static parameters
            starts_vec = param_->starts;
            ends_vec = param_->ends;
            axes_vec = param_->axes;
            steps_vec = param_->steps;
        } else {
            // Read from input tensors (dynamic slice)
            // inputs[0] = data, inputs[1] = starts, inputs[2] = ends,
            // inputs[3] = axes (optional), inputs[4] = steps (optional)
            if (inputs.size() < 3) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }

            if (inputs[1]) starts_vec = read_int64_tensor(inputs[1]);
            if (inputs[2]) ends_vec = read_int64_tensor(inputs[2]);
            if (inputs.size() > 3 && inputs[3]) axes_vec = read_int64_tensor(inputs[3]);
            if (inputs.size() > 4 && inputs[4]) steps_vec = read_int64_tensor(inputs[4]);

            if (starts_vec.empty() || ends_vec.empty()) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
        }

        // Normalize axes
        if (axes_vec.empty()) {
            for (int64_t i = 0; i < static_cast<int64_t>(starts_vec.size()); ++i) {
                axes_vec.push_back(i);
            }
        }
        for (auto& ax : axes_vec) {
            if (ax < 0) ax += ndim;
        }

        // Normalize steps
        if (steps_vec.empty()) {
            steps_vec.resize(axes_vec.size(), 1);
        }

        // Build per-axis slice info
        std::vector<int64_t> starts(ndim, 0);
        std::vector<int64_t> slice_steps(ndim, 1);

        for (size_t i = 0; i < axes_vec.size(); ++i) {
            int64_t axis = axes_vec[i];
            if (axis < 0 || axis >= ndim) continue;

            if (i >= starts_vec.size()) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }

            int64_t dim_size = input_dims[axis];
            int64_t start = starts_vec[i];
            int64_t step = steps_vec[i];

            if (start < 0) start += dim_size;
            start = std::max(int64_t(0), std::min(start, dim_size));

            starts[axis] = start;
            slice_steps[axis] = step;
        }

        // Compute input strides
        std::vector<int64_t> in_strides(ndim);
        int64_t stride = 1;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            in_strides[i] = stride;
            stride *= input_dims[i];
        }

        // Compute output strides
        std::vector<int64_t> out_strides(ndim);
        stride = 1;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            out_strides[i] = stride;
            stride *= output_dims[i];
        }

        const float* data_in = static_cast<const float*>(input->data());
        float* data_out = static_cast<float*>(output->data());
        int64_t out_total = output_shape.numel();

        // Iterate over output elements
        for (int64_t out_idx = 0; out_idx < out_total; ++out_idx) {
            std::vector<int64_t> out_coords(ndim);
            int64_t remaining = out_idx;
            for (int64_t d = 0; d < ndim; ++d) {
                out_coords[d] = remaining / out_strides[d];
                remaining %= out_strides[d];
            }

            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t in_coord = starts[d] + out_coords[d] * slice_steps[d];
                in_idx += in_coord * in_strides[d];
            }

            data_out[out_idx] = data_in[in_idx];
        }

        return core::Status::SUCCESS;
    }

private:
    std::shared_ptr<SliceParam> param_;
};

REGISTER_PLUGIN_SIMPLE(SliceCPUPlugin, "Slice", kSLICE, CPU)

}  // namespace operators
}  // namespace mini_infer
