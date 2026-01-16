#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>
#include <numeric>

namespace mini_infer {
namespace operators {

/**
 * @brief Transpose CPU Plugin
 *
 * Permutes the dimensions of the input tensor according to perm.
 */
class TransposeCPUPlugin : public CPUPlugin<TransposeCPUPlugin, TransposeParam> {
public:
    TransposeCPUPlugin() {
        param_ = std::make_shared<TransposeParam>();
    }
    ~TransposeCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Transpose";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kTRANSPOSE;
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
        const size_t ndim = input_shape.ndim();

        // Get permutation
        std::vector<int64_t> perm;
        if (param_ && !param_->perm.empty()) {
            perm = param_->perm;
        } else {
            // Default: reverse dimensions
            perm.resize(ndim);
            for (size_t i = 0; i < ndim; ++i) {
                perm[i] = static_cast<int64_t>(ndim - 1 - i);
            }
        }

        if (perm.size() != ndim) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Compute output shape
        std::vector<int64_t> output_dims(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            int64_t axis = perm[i];
            if (axis < 0) {
                axis += static_cast<int64_t>(ndim);
            }
            if (axis < 0 || axis >= static_cast<int64_t>(ndim)) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
            output_dims[i] = input_shape[axis];
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

        if (input->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const core::Shape& in_shape = input->shape();
        const core::Shape& out_shape = output->shape();
        const size_t ndim = in_shape.ndim();

        // Get permutation
        std::vector<int64_t> perm;
        if (param_ && !param_->perm.empty()) {
            perm = param_->perm;
        } else {
            perm.resize(ndim);
            for (size_t i = 0; i < ndim; ++i) {
                perm[i] = static_cast<int64_t>(ndim - 1 - i);
            }
        }

        // Compute input strides
        std::vector<int64_t> in_strides(ndim);
        int64_t stride = 1;
        for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
            in_strides[i] = stride;
            stride *= in_shape[i];
        }

        // Compute output strides
        std::vector<int64_t> out_strides(ndim);
        stride = 1;
        for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
            out_strides[i] = stride;
            stride *= out_shape[i];
        }

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        const int64_t total = out_shape.numel();
        std::vector<int64_t> out_indices(ndim, 0);

        for (int64_t i = 0; i < total; ++i) {
            // Compute input index from output index using permutation
            int64_t in_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                int64_t axis = perm[d];
                if (axis < 0) axis += static_cast<int64_t>(ndim);
                in_idx += out_indices[d] * in_strides[axis];
            }

            out_data[i] = in_data[in_idx];

            // Increment output indices
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                out_indices[d]++;
                if (out_indices[d] < out_shape[d]) {
                    break;
                }
                out_indices[d] = 0;
            }
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(TransposeCPUPlugin, "Transpose", kTRANSPOSE, CPU)

}  // namespace operators
}  // namespace mini_infer
