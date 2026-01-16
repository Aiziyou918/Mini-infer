#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <cstring>

namespace mini_infer {
namespace operators {

/**
 * @brief Gather CPU Plugin
 *
 * Gathers elements from data along the specified axis using indices.
 * output_shape = data_shape[:axis] + indices_shape + data_shape[axis+1:]
 */
class GatherCPUPlugin : public CPUPlugin<GatherCPUPlugin, GatherParam> {
public:
    GatherCPUPlugin() {
        param_ = std::make_shared<GatherParam>();
    }
    ~GatherCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Gather";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kGATHER;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return 2;  // data, indices
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {

        if (input_shapes.size() != 2) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& data_shape = input_shapes[0];
        const auto& indices_shape = input_shapes[1];
        const int64_t data_ndim = static_cast<int64_t>(data_shape.ndim());

        // Handle empty data shape (undefined)
        if (data_ndim == 0) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Get axis
        int64_t axis = param_ ? param_->axis : 0;
        if (axis < 0) {
            axis += data_ndim;
        }
        if (axis < 0 || axis >= data_ndim) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Build output shape: data_shape[:axis] + indices_shape + data_shape[axis+1:]
        std::vector<int64_t> output_dims;

        // Add dimensions before axis
        for (int64_t i = 0; i < axis; ++i) {
            output_dims.push_back(data_shape[i]);
        }

        // Add indices dimensions (if indices is scalar, this adds nothing)
        for (size_t i = 0; i < indices_shape.ndim(); ++i) {
            output_dims.push_back(indices_shape[i]);
        }

        // Add dimensions after axis
        for (int64_t i = axis + 1; i < data_ndim; ++i) {
            output_dims.push_back(data_shape[i]);
        }

        // Note: if output_dims is empty, the output is a scalar (0-dim tensor)
        // This is correct per ONNX spec when indices is scalar and data is 1D

        output_shapes.clear();
        output_shapes.push_back(core::Shape(output_dims));
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;

        if (inputs.size() != 2 || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& data = inputs[0];
        const auto& indices = inputs[1];
        auto& output = outputs[0];

        if (!data || !indices || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const core::DataType data_dtype = data->dtype();
        if (data_dtype != core::DataType::FLOAT32 &&
            data_dtype != core::DataType::INT64 &&
            data_dtype != core::DataType::INT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const core::Shape& data_shape = data->shape();
        const core::Shape& indices_shape = indices->shape();
        const core::Shape& out_shape = output->shape();
        const int64_t data_ndim = static_cast<int64_t>(data_shape.ndim());

        // Get axis
        int64_t axis = param_ ? param_->axis : 0;
        if (axis < 0) {
            axis += data_ndim;
        }

        // Compute sizes
        int64_t outer_size = 1;
        for (int64_t i = 0; i < axis; ++i) {
            outer_size *= data_shape[i];
        }

        const int64_t axis_size = data_shape[axis];

        int64_t inner_size = 1;
        for (int64_t i = axis + 1; i < data_ndim; ++i) {
            inner_size *= data_shape[i];
        }

        const int64_t indices_total = indices_shape.numel();

        // Get indices data (support both INT32 and INT64)
        std::vector<int64_t> indices_vec(indices_total);
        if (indices->dtype() == core::DataType::INT64) {
            const int64_t* idx_data = static_cast<const int64_t*>(indices->data());
            for (int64_t i = 0; i < indices_total; ++i) {
                indices_vec[i] = idx_data[i];
            }
        } else if (indices->dtype() == core::DataType::INT32) {
            const int32_t* idx_data = static_cast<const int32_t*>(indices->data());
            for (int64_t i = 0; i < indices_total; ++i) {
                indices_vec[i] = static_cast<int64_t>(idx_data[i]);
            }
        } else {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        // Template-like gather for different data types
        auto do_gather = [&](auto* data_ptr, auto* out_ptr) -> core::Status {
            for (int64_t o = 0; o < outer_size; ++o) {
                for (int64_t idx_i = 0; idx_i < indices_total; ++idx_i) {
                    int64_t index = indices_vec[idx_i];

                    // Handle negative indices
                    if (index < 0) {
                        index += axis_size;
                    }

                    // Bounds check
                    if (index < 0 || index >= axis_size) {
                        return core::Status::ERROR_INVALID_ARGUMENT;
                    }

                    // Copy inner slice
                    const auto* src = data_ptr + (o * axis_size + index) * inner_size;
                    auto* dst = out_ptr + (o * indices_total + idx_i) * inner_size;

                    std::memcpy(dst, src, static_cast<size_t>(inner_size) * sizeof(*data_ptr));
                }
            }
            return core::Status::SUCCESS;
        };

        if (data_dtype == core::DataType::FLOAT32) {
            return do_gather(static_cast<const float*>(data->data()),
                           static_cast<float*>(output->data()));
        } else if (data_dtype == core::DataType::INT64) {
            return do_gather(static_cast<const int64_t*>(data->data()),
                           static_cast<int64_t*>(output->data()));
        } else {
            return do_gather(static_cast<const int32_t*>(data->data()),
                           static_cast<int32_t*>(output->data()));
        }
    }
};

REGISTER_PLUGIN_SIMPLE(GatherCPUPlugin, "Gather", kGATHER, CPU)

}  // namespace operators
}  // namespace mini_infer
