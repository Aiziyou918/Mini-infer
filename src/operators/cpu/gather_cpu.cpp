#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <cstring>

namespace mini_infer {
namespace operators {

/**
 * @brief Gather CPU Plugin
 *
 * Implements gather operation for embedding lookup.
 * output = data[indices] along specified axis
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

        const auto& data_dims = input_shapes[0].dims();
        const auto& indices_dims = input_shapes[1].dims();

        int64_t axis = param_ ? param_->axis : 0;
        int64_t data_ndim = static_cast<int64_t>(data_dims.size());
        if (axis < 0) axis += data_ndim;
        if (axis < 0 || axis >= data_ndim) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Output shape: data_dims[:axis] + indices_dims + data_dims[axis+1:]
        std::vector<int64_t> out_dims;
        for (int64_t i = 0; i < axis; ++i) {
            out_dims.push_back(data_dims[i]);
        }
        for (const auto& d : indices_dims) {
            out_dims.push_back(d);
        }
        for (int64_t i = axis + 1; i < data_ndim; ++i) {
            out_dims.push_back(data_dims[i]);
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

        if (inputs.size() != 2 || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& data = inputs[0];
        const auto& indices = inputs[1];
        auto& output = outputs[0];

        if (!data || !indices || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        if (data->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& data_dims = data->shape().dims();
        const int64_t data_ndim = static_cast<int64_t>(data_dims.size());

        int64_t axis = param_ ? param_->axis : 0;
        if (axis < 0) axis += data_ndim;

        // Compute sizes
        int64_t outer_size = 1;
        int64_t inner_size = 1;
        for (int64_t i = 0; i < axis; ++i) {
            outer_size *= data_dims[i];
        }
        for (int64_t i = axis + 1; i < data_ndim; ++i) {
            inner_size *= data_dims[i];
        }

        const int64_t axis_size = data_dims[axis];
        const int64_t num_indices = indices->shape().numel();

        const float* data_ptr = static_cast<const float*>(data->data());
        float* out_ptr = static_cast<float*>(output->data());

        // Handle different index types
        if (indices->dtype() == core::DataType::INT64) {
            const int64_t* idx_ptr = static_cast<const int64_t*>(indices->data());

            for (int64_t o = 0; o < outer_size; ++o) {
                for (int64_t i = 0; i < num_indices; ++i) {
                    int64_t idx = idx_ptr[i];
                    if (idx < 0) idx += axis_size;

                    const float* src = data_ptr + (o * axis_size + idx) * inner_size;
                    float* dst = out_ptr + (o * num_indices + i) * inner_size;
                    std::memcpy(dst, src, inner_size * sizeof(float));
                }
            }
        } else if (indices->dtype() == core::DataType::INT32) {
            const int32_t* idx_ptr = static_cast<const int32_t*>(indices->data());

            for (int64_t o = 0; o < outer_size; ++o) {
                for (int64_t i = 0; i < num_indices; ++i) {
                    int64_t idx = idx_ptr[i];
                    if (idx < 0) idx += axis_size;

                    const float* src = data_ptr + (o * axis_size + idx) * inner_size;
                    float* dst = out_ptr + (o * num_indices + i) * inner_size;
                    std::memcpy(dst, src, inner_size * sizeof(float));
                }
            }
        } else {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(GatherCPUPlugin, "Gather", kGATHER, CPU)

}  // namespace operators
}  // namespace mini_infer
