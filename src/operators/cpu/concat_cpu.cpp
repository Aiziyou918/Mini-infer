#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>
#include <cstring>

namespace mini_infer {
namespace operators {

/**
 * @brief Concat CPU Plugin
 *
 * Concatenates tensors along a specified axis.
 * output = concat([input_0, input_1, ...], axis)
 */
class ConcatCPUPlugin : public CPUPlugin<ConcatCPUPlugin, ConcatParam> {
public:
    ConcatCPUPlugin() {
        param_ = std::make_shared<ConcatParam>();
    }
    ~ConcatCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Concat";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kCONCAT;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return -1;  // Variable number of inputs
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {
        if (input_shapes.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        int64_t axis = param_ ? param_->axis : 0;

        // Special case: concatenating scalars or 1D tensors along axis 0
        // This is common in dynamic shape computation (e.g., building reshape target)
        bool all_scalar_or_1d = true;
        int64_t total_elements = 0;
        for (const auto& shape : input_shapes) {
            size_t ndim = shape.ndim();
            if (ndim == 0) {
                // Scalar contributes 1 element
                total_elements += 1;
            } else if (ndim == 1) {
                int64_t dim = shape.dims()[0];
                if (dim == -1) {
                    total_elements = -1;  // Dynamic
                    break;
                }
                total_elements += dim;
            } else {
                all_scalar_or_1d = false;
                break;
            }
        }

        if (all_scalar_or_1d && axis == 0) {
            // Output is 1D tensor with total_elements
            output_shapes.clear();
            output_shapes.push_back(core::Shape({total_elements}));
            return core::Status::SUCCESS;
        }

        // General case: all inputs must have same ndim
        const auto& first_dims = input_shapes[0].dims();
        int64_t ndim = static_cast<int64_t>(first_dims.size());

        if (axis < 0) axis += ndim;
        if (ndim == 0 || axis < 0 || axis >= ndim) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Compute output shape
        std::vector<int64_t> out_dims = first_dims;
        int64_t concat_dim = first_dims[axis];

        for (size_t i = 1; i < input_shapes.size(); ++i) {
            const auto& dims = input_shapes[i].dims();
            if (dims.size() != first_dims.size()) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }

            // Check that all dims except axis match
            for (size_t d = 0; d < dims.size(); ++d) {
                if (d == static_cast<size_t>(axis)) {
              // Handle dynamic dimensions
                    if (dims[d] == -1 || concat_dim == -1) {
                        concat_dim = -1;
                    } else {
                        concat_dim += dims[d];
                    }
                } else {
                    // Non-concat dimensions must match (or be dynamic)
                    if (dims[d] != first_dims[d] && dims[d] != -1 && first_dims[d] != -1) {
                        return core::Status::ERROR_INVALID_ARGUMENT;
                    }
                    // Propagate concrete dimension if available
                    if (out_dims[d] == -1 && dims[d] != -1) {
                        out_dims[d] = dims[d];
                    }
                }
            }
        }

        out_dims[axis] = concat_dim;
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

        auto& output = outputs[0];
        if (!output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Get axis
        const auto& first_shape = inputs[0]->shape();
        int64_t ndim = static_cast<int64_t>(first_shape.ndim());
        int64_t axis = param_ ? param_->axis : 0;
        if (axis < 0) axis += ndim;

        const auto& out_dims = output->shape().dims();

        // Compute outer and inner sizes
        int64_t outer_size = 1;
        int64_t inner_size = 1;
        for (int64_t i = 0; i < axis; ++i) {
            outer_size *= out_dims[i];
        }
        for (int64_t i = axis + 1; i < ndim; ++i) {
            inner_size *= out_dims[i];
        }

        // Handle different data types
        if (inputs[0]->dtype() == core::DataType::FLOAT32) {
            return concat_impl<float>(inputs, output, axis, outer_size, inner_size);
        } else if (inputs[0]->dtype() == core::DataType::INT64) {
            return concat_impl<int64_t>(inputs, output, axis, outer_size, inner_size);
        } else if (inputs[0]->dtype() == core::DataType::INT32) {
            return concat_impl<int32_t>(inputs, output, axis, outer_size, inner_size);
        }

        return core::Status::ERROR_NOT_IMPLEMENTED;
    }

private:
    template<typename T>
    core::Status concat_impl(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::shared_ptr<core::Tensor>& output,
        int64_t axis,
        int64_t outer_size,
        int64_t inner_size) {

        T* out_ptr = static_cast<T*>(output->data());
        int64_t out_axis_stride = output->shape().dims()[axis] * inner_size;

        for (int64_t o = 0; o < outer_size; ++o) {
            int64_t axis_offset = 0;
            for (const auto& input : inputs) {
                if (!input) continue;

                const T* in_ptr = static_cast<const T*>(input->data());
                int64_t in_axis_size = input->shape().dims()[axis];
                int64_t copy_size = in_axis_size * inner_size;

                // Copy this input's slice
                const T* src = in_ptr + o * copy_size;
                T* dst = out_ptr + o * out_axis_stride + axis_offset * inner_size;
                std::memcpy(dst, src, copy_size * sizeof(T));

                axis_offset += in_axis_size;
            }
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(ConcatCPUPlugin, "Concat", kCONCAT, CPU)

}  // namespace operators
}  // namespace mini_infer
