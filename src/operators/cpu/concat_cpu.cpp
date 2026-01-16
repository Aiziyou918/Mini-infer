#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <cstring>
#include <type_traits>

namespace mini_infer {
namespace operators {

/**
 * @brief Concat CPU Plugin
 *
 * Concatenates tensors along a specified axis.
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

        // First pass: find max ndim among all inputs
        size_t max_ndim = 0;
        for (const auto& shape : input_shapes) {
            if (shape.ndim() > max_ndim) {
                max_ndim = shape.ndim();
            }
        }
        if (max_ndim == 0) {
            max_ndim = 1;
        }
        const int64_t ndim = static_cast<int64_t>(max_ndim);

        // Handle axis (support negative indexing)
        int64_t axis = param_ ? param_->axis : 0;
        if (axis < 0) {
            axis += ndim;
        }
        if (axis < 0 || axis >= ndim) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Second pass: broadcast each input to max_ndim and accumulate
        std::vector<int64_t> output_dims(max_ndim, 1);
        int64_t concat_dim = 0;
        bool dims_initialized = false;

        for (const auto& shape : input_shapes) {
            // Broadcast shape to max_ndim by prepending 1s
            std::vector<int64_t> broadcasted(max_ndim, 1);
            if (shape.ndim() > 0) {
                size_t offset = max_ndim - shape.ndim();
                for (size_t d = 0; d < shape.ndim(); ++d) {
                    broadcasted[offset + d] = shape[d];
                }
            }

            if (!dims_initialized) {
                for (int64_t d = 0; d < ndim; ++d) {
                    if (d != axis) {
                        output_dims[d] = broadcasted[d];
                    }
                }
                dims_initialized = true;
            } else {
                // Verify non-axis dimensions consistency
                for (int64_t d = 0; d < ndim; ++d) {
                    if (d != axis && broadcasted[d] != output_dims[d]) {
                        return core::Status::ERROR_INVALID_ARGUMENT;
                    }
                }
            }

            // Accumulate concat axis
            concat_dim += broadcasted[axis];
        }

        output_dims[axis] = concat_dim;
        output_shapes.clear();
        output_shapes.push_back(core::Shape(output_dims));
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;

        if (inputs.empty() || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        auto& output = outputs[0];
        if (!output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const core::Shape& out_shape = output->shape();
        const int64_t ndim = static_cast<int64_t>(out_shape.ndim());

        // Get axis
        int64_t axis = param_ ? param_->axis : 0;
        if (axis < 0) {
            axis += ndim;
        }

        // Compute outer and inner sizes
        int64_t outer_size = 1;
        for (int64_t d = 0; d < axis; ++d) {
            outer_size *= out_shape[d];
        }

        int64_t inner_size = 1;
        for (int64_t d = axis + 1; d < ndim; ++d) {
            inner_size *= out_shape[d];
        }

        if (inputs[0]->dtype() != output->dtype()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto dtype = output->dtype();
        if (dtype != core::DataType::FLOAT32 &&
            dtype != core::DataType::INT64 &&
            dtype != core::DataType::INT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        auto copy_concat = [&](auto* out_ptr) -> core::Status {
            int64_t axis_offset = 0;
            for (const auto& input : inputs) {
                if (!input) {
                    return core::Status::ERROR_INVALID_ARGUMENT;
                }

                if (input->dtype() != dtype) {
                    return core::Status::ERROR_INVALID_ARGUMENT;
                }

                const auto* in_data = static_cast<const std::decay_t<decltype(*out_ptr)>*>(
                    input->data());
                const core::Shape& in_shape = input->shape();
                int64_t in_axis_size = 1;
                if (in_shape.ndim() > 0) {
                    in_axis_size = in_shape[axis];
                } else if (axis != 0) {
                    return core::Status::ERROR_INVALID_ARGUMENT;
                }

                for (int64_t o = 0; o < outer_size; ++o) {
                    const int64_t out_offset = o * out_shape[axis] * inner_size +
                                               axis_offset * inner_size;
                    const int64_t in_offset = o * in_axis_size * inner_size;
                    const size_t copy_size =
                        static_cast<size_t>(in_axis_size * inner_size) *
                        sizeof(std::decay_t<decltype(*out_ptr)>);

                    std::memcpy(out_ptr + out_offset, in_data + in_offset, copy_size);
                }

                axis_offset += in_axis_size;
            }

            return core::Status::SUCCESS;
        };
        if (dtype == core::DataType::FLOAT32) {
            return copy_concat(static_cast<float*>(output->data()));
        }
        if (dtype == core::DataType::INT64) {
            return copy_concat(static_cast<int64_t*>(output->data()));
        }
        return copy_concat(static_cast<int32_t*>(output->data()));
    }
};

REGISTER_PLUGIN_SIMPLE(ConcatCPUPlugin, "Concat", kCONCAT, CPU)

}  // namespace operators
}  // namespace mini_infer
