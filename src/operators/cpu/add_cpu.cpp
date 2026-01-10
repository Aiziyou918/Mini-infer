#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>
#include <cstdint>

namespace mini_infer {
namespace operators {

namespace {

/**
 * @brief Compute broadcast shape from two input shapes
 * @return true if broadcast is valid, false otherwise
 *
 * Handles dynamic dimensions (-1) according to ONNX semantics:
 * - If both dims are concrete and equal, output is that value
 * - If one dim is 1, output is the other dim (broadcast)
 * - If one dim is -1 (dynamic), output is -1 (remains dynamic)
 * - If both dims are -1, output is -1
 */
bool compute_broadcast_shape(
    const core::Shape& a,
    const core::Shape& b,
    core::Shape& result
) {
    const auto& dims_a = a.dims();
    const auto& dims_b = b.dims();

    size_t ndim_a = dims_a.size();
    size_t ndim_b = dims_b.size();
    size_t ndim_out = std::max(ndim_a, ndim_b);

    std::vector<int64_t> out_dims(ndim_out);

    for (size_t i = 0; i < ndim_out; ++i) {
        int64_t dim_a = (i < ndim_out - ndim_a) ? 1 : dims_a[i - (ndim_out - ndim_a)];
        int64_t dim_b = (i < ndim_out - ndim_b) ? 1 : dims_b[i - (ndim_out - ndim_b)];

        // Handle dynamic dimensions (-1)
        if (dim_a == -1 || dim_b == -1) {
            // If either is dynamic, result is dynamic (unless one is 1)
            if (dim_a == 1) {
                out_dims[i] = dim_b;
            } else if (dim_b == 1) {
                out_dims[i] = dim_a;
            } else {
                out_dims[i] = -1;  // Dynamic output
            }
        } else if (dim_a == dim_b) {
            out_dims[i] = dim_a;
        } else if (dim_a == 1) {
            out_dims[i] = dim_b;
        } else if (dim_b == 1) {
            out_dims[i] = dim_a;
        } else {
            return false;  // Incompatible shapes
        }
    }

    result = core::Shape(out_dims);
    return true;
}

/**
 * @brief Compute strides for broadcast iteration
 */
std::vector<int64_t> compute_broadcast_strides(
    const core::Shape& shape,
    const core::Shape& broadcast_shape
) {
    const auto& dims = shape.dims();
    const auto& bc_dims = broadcast_shape.dims();

    size_t ndim = dims.size();
    size_t bc_ndim = bc_dims.size();

    std::vector<int64_t> strides(bc_ndim, 0);

    // Compute original strides
    std::vector<int64_t> orig_strides(ndim);
    int64_t stride = 1;
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
        orig_strides[i] = stride;
        stride *= dims[i];
    }

    // Map to broadcast strides (0 for broadcast dimensions)
    for (size_t i = 0; i < bc_ndim; ++i) {
        size_t orig_idx = i - (bc_ndim - ndim);
        if (i >= bc_ndim - ndim) {
            if (dims[orig_idx] == bc_dims[i]) {
                strides[i] = orig_strides[orig_idx];
            } else {
                strides[i] = 0;  // Broadcast dimension
            }
        }
    }

    return strides;
}

/**
 * @brief Convert linear index to multi-dimensional index
 */
void linear_to_multi_index(
    int64_t linear_idx,
    const std::vector<int64_t>& dims,
    std::vector<int64_t>& multi_idx
) {
    size_t ndim = dims.size();
    multi_idx.resize(ndim);

    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
        multi_idx[i] = linear_idx % dims[i];
        linear_idx /= dims[i];
    }
}

/**
 * @brief Compute linear index from multi-dimensional index and strides
 */
int64_t multi_to_linear_index(
    const std::vector<int64_t>& multi_idx,
    const std::vector<int64_t>& strides
) {
    int64_t idx = 0;
    for (size_t i = 0; i < multi_idx.size(); ++i) {
        idx += multi_idx[i] * strides[i];
    }
    return idx;
}

}  // namespace

/**
 * @brief Add CPU Plugin
 *
 * Implements element-wise addition with NumPy-style broadcasting.
 * output = input_a + input_b
 */
class AddCPUPlugin : public SimpleCPUPlugin<AddCPUPlugin> {
public:
    AddCPUPlugin() = default;
    ~AddCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Add";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kADD;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return 2;
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {
        if (input_shapes.size() != 2) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        output_shapes.clear();
        core::Shape out_shape;
        if (!compute_broadcast_shape(input_shapes[0], input_shapes[1], out_shape)) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
        output_shapes.push_back(out_shape);
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

        const auto& input_a = inputs[0];
        const auto& input_b = inputs[1];
        auto& output = outputs[0];

        if (!input_a || !input_b || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        if (input_a->dtype() != core::DataType::FLOAT32 ||
            input_b->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& shape_a = input_a->shape();
        const auto& shape_b = input_b->shape();
        const auto& out_shape = output->shape();

        const float* data_a = static_cast<const float*>(input_a->data());
        const float* data_b = static_cast<const float*>(input_b->data());
        float* data_out = static_cast<float*>(output->data());

        const int64_t total = out_shape.numel();

        // Fast path: same shape (no broadcast needed)
        if (shape_a == shape_b) {
            for (int64_t i = 0; i < total; ++i) {
                data_out[i] = data_a[i] + data_b[i];
            }
            return core::Status::SUCCESS;
        }

        // General broadcast path
        auto strides_a = compute_broadcast_strides(shape_a, out_shape);
        auto strides_b = compute_broadcast_strides(shape_b, out_shape);

        const auto& out_dims = out_shape.dims();
        std::vector<int64_t> multi_idx;

        for (int64_t i = 0; i < total; ++i) {
            linear_to_multi_index(i, out_dims, multi_idx);
            int64_t idx_a = multi_to_linear_index(multi_idx, strides_a);
            int64_t idx_b = multi_to_linear_index(multi_idx, strides_b);
            data_out[i] = data_a[idx_a] + data_b[idx_b];
        }

        return core::Status::SUCCESS;
    }
};

// Define creator and register plugin
REGISTER_PLUGIN_SIMPLE(AddCPUPlugin, "Add", kADD, CPU)

}  // namespace operators
}  // namespace mini_infer
