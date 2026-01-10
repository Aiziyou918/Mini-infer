#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>
#include <numeric>

namespace mini_infer {
namespace operators {

/**
 * @brief Transpose CPU Plugin
 *
 * Implements tensor transpose/permutation.
 * output = input.permute(perm)
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
        if (input_shapes.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& in_shape = input_shapes[0];
        const auto& in_dims = in_shape.dims();
        size_t ndim = in_dims.size();

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
        std::vector<int64_t> out_dims(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            int64_t axis = perm[i];
            if (axis < 0) axis += static_cast<int64_t>(ndim);
            if (axis < 0 || axis >= static_cast<int64_t>(ndim)) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
            out_dims[i] = in_dims[axis];
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

        const auto& in_shape = input->shape();
        const auto& in_dims = in_shape.dims();
        size_t ndim = in_dims.size();

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

        // Normalize negative axes
        for (auto& p : perm) {
            if (p < 0) p += static_cast<int64_t>(ndim);
        }

        // Compute input strides
        std::vector<int64_t> in_strides(ndim);
        int64_t stride = 1;
        for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
            in_strides[i] = stride;
            stride *= in_dims[i];
        }

        // Compute output strides
        const auto& out_dims = output->shape().dims();
        std::vector<int64_t> out_strides(ndim);
        stride = 1;
        for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
            out_strides[i] = stride;
            stride *= out_dims[i];
        }

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        const int64_t total = in_shape.numel();

        // Transpose using index mapping
        for (int64_t out_idx = 0; out_idx < total; ++out_idx) {
            // Convert output linear index to multi-dimensional index
            std::vector<int64_t> out_coords(ndim);
            int64_t remaining = out_idx;
            for (size_t d = 0; d < ndim; ++d) {
                out_coords[d] = remaining / out_strides[d];
                remaining %= out_strides[d];
            }

            // Map to input coordinates using inverse permutation
            int64_t in_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                in_idx += out_coords[d] * in_strides[perm[d]];
            }

            out_data[out_idx] = in_data[in_idx];
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(TransposeCPUPlugin, "Transpose", kTRANSPOSE, CPU)

}  // namespace operators
}  // namespace mini_infer
