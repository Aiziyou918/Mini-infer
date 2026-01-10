#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>
#include <numeric>
#include <cstdint>

namespace mini_infer {
namespace operators {

class ReduceMeanCPUPlugin : public SimpleCPUPlugin<ReduceMeanCPUPlugin> {
public:
    ReduceMeanCPUPlugin() = default;
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

    void set_param(std::shared_ptr<PluginParam> param) override {
        param_ = std::dynamic_pointer_cast<ReduceMeanParam>(param);
    }

    const void* get_param_ptr() const noexcept override {
        return param_.get();
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {
        if (input_shapes.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_dims = input_shapes[0].dims();
        int64_t ndim = static_cast<int64_t>(input_dims.size());

        std::vector<int64_t> axes;
        if (param_ && !param_->axes.empty()) {
            axes = param_->axes;
            // Normalize negative axes
            for (auto& ax : axes) {
                if (ax < 0) ax += ndim;
            }
        } else {
            // Default: reduce all axes
            for (int64_t i = 0; i < ndim; ++i) {
                axes.push_back(i);
            }
        }

        bool keepdims = param_ ? param_->keepdims : true;

        std::vector<int64_t> out_dims;
        for (int64_t i = 0; i < ndim; ++i) {
            bool is_reduce_axis = std::find(axes.begin(), axes.end(), i) != axes.end();
            if (is_reduce_axis) {
                if (keepdims) {
                    out_dims.push_back(1);
                }
            } else {
                out_dims.push_back(input_dims[i]);
            }
        }

        if (out_dims.empty()) {
            out_dims.push_back(1);  // Scalar output
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

        if (input->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& input_shape = input->shape();
        const auto& input_dims = input_shape.dims();
        int64_t ndim = static_cast<int64_t>(input_dims.size());

        std::vector<int64_t> axes;
        if (param_ && !param_->axes.empty()) {
            axes = param_->axes;
            for (auto& ax : axes) {
                if (ax < 0) ax += ndim;
            }
        } else {
            for (int64_t i = 0; i < ndim; ++i) {
                axes.push_back(i);
            }
        }

        std::sort(axes.begin(), axes.end());

        const float* data_in = static_cast<const float*>(input->data());
        float* data_out = static_cast<float*>(output->data());

        // Compute strides for input
        std::vector<int64_t> in_strides(ndim);
        int64_t stride = 1;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            in_strides[i] = stride;
            stride *= input_dims[i];
        }

        // Compute output shape and strides
        const auto& out_shape = output->shape();
        const auto& out_dims = out_shape.dims();
        int64_t out_ndim = static_cast<int64_t>(out_dims.size());

        std::vector<int64_t> out_strides(out_ndim);
        stride = 1;
        for (int64_t i = out_ndim - 1; i >= 0; --i) {
            out_strides[i] = stride;
            stride *= out_dims[i];
        }

        int64_t out_total = out_shape.numel();
        int64_t in_total = input_shape.numel();

        // Initialize output to zero
        std::fill(data_out, data_out + out_total, 0.0f);

        // Compute reduction count
        int64_t reduce_count = 1;
        for (auto ax : axes) {
            reduce_count *= input_dims[ax];
        }

        // Map input index to output index and accumulate
        bool keepdims = param_ ? param_->keepdims : true;

        for (int64_t in_idx = 0; in_idx < in_total; ++in_idx) {
            // Convert linear index to multi-dimensional index
            std::vector<int64_t> in_coords(ndim);
            int64_t remaining = in_idx;
            for (int64_t d = 0; d < ndim; ++d) {
                in_coords[d] = remaining / in_strides[d];
                remaining %= in_strides[d];
            }

            // Compute output index
            std::vector<int64_t> out_coords;
            for (int64_t d = 0; d < ndim; ++d) {
                bool is_reduce_axis = std::find(axes.begin(), axes.end(), d) != axes.end();
                if (is_reduce_axis) {
                    if (keepdims) {
                        out_coords.push_back(0);
                    }
                } else {
                    out_coords.push_back(in_coords[d]);
                }
            }

            if (out_coords.empty()) {
                out_coords.push_back(0);
            }

            int64_t out_idx = 0;
            for (size_t d = 0; d < out_coords.size(); ++d) {
                out_idx += out_coords[d] * out_strides[d];
            }

            data_out[out_idx] += data_in[in_idx];
        }

        // Divide by count to get mean
        for (int64_t i = 0; i < out_total; ++i) {
            data_out[i] /= static_cast<float>(reduce_count);
        }

        return core::Status::SUCCESS;
    }

private:
    std::shared_ptr<ReduceMeanParam> param_;
};

REGISTER_PLUGIN_SIMPLE(ReduceMeanCPUPlugin, "ReduceMean", kREDUCE_MEAN, CPU)

}  // namespace operators
}  // namespace mini_infer
