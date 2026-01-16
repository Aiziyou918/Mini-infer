#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>
#include <cmath>
#include <functional>

namespace mini_infer {
namespace operators {

namespace {

/**
 * @brief Compute broadcast output shape from two input shapes
 *
 * Follows NumPy broadcasting rules:
 * - Align shapes from the right
 * - Dimensions are compatible if they are equal or one of them is 1
 */
core::Status broadcast_shapes(const core::Shape& a, const core::Shape& b,
                               core::Shape& output) {
    const size_t ndim_a = a.ndim();
    const size_t ndim_b = b.ndim();
    const size_t ndim_out = std::max(ndim_a, ndim_b);

    std::vector<int64_t> out_dims(ndim_out);

    for (size_t i = 0; i < ndim_out; ++i) {
        // Get dimensions from right to left
        const int64_t dim_a = (i < ndim_a) ? a[ndim_a - 1 - i] : 1;
        const int64_t dim_b = (i < ndim_b) ? b[ndim_b - 1 - i] : 1;

        if (dim_a == dim_b) {
            out_dims[ndim_out - 1 - i] = dim_a;
        } else if (dim_a == 1) {
            out_dims[ndim_out - 1 - i] = dim_b;
        } else if (dim_b == 1) {
            out_dims[ndim_out - 1 - i] = dim_a;
        } else {
            // Incompatible dimensions
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }

    output = core::Shape(out_dims);
    return core::Status::SUCCESS;
}

/**
 * @brief Compute strides for broadcasting
 */
void compute_broadcast_strides(const core::Shape& shape, const core::Shape& target,
                                std::vector<int64_t>& strides) {
    const size_t ndim = target.ndim();
    const size_t shape_ndim = shape.ndim();
    strides.resize(ndim);

    int64_t stride = 1;
    for (size_t i = 0; i < ndim; ++i) {
        const size_t idx = ndim - 1 - i;
        const size_t shape_idx = shape_ndim - 1 - i;

        if (i < shape_ndim && shape[shape_idx] == target[idx]) {
            strides[idx] = stride;
            stride *= shape[shape_idx];
        } else {
            // Broadcasting dimension (size 1 or missing)
            strides[idx] = 0;
        }
    }
}

/**
 * @brief Generic binary elementwise operation with broadcasting
 */
template <typename Op>
core::Status elementwise_binary_op(
    const std::vector<std::shared_ptr<core::Tensor>>& inputs,
    std::vector<std::shared_ptr<core::Tensor>>& outputs,
    Op op) {

    if (inputs.size() != 2 || outputs.size() != 1) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    const auto& a = inputs[0];
    const auto& b = inputs[1];
    auto& out = outputs[0];

    if (!a || !b || !out) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    const core::DataType dtype = a->dtype();
    if (dtype != b->dtype()) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    const core::Shape& out_shape = out->shape();
    const int64_t total = out_shape.numel();

    // Compute broadcast strides
    std::vector<int64_t> a_strides, b_strides;
    compute_broadcast_strides(a->shape(), out_shape, a_strides);
    compute_broadcast_strides(b->shape(), out_shape, b_strides);

    // Compute output indices and map to input indices
    const size_t ndim = out_shape.ndim();
    std::vector<int64_t> indices(ndim, 0);

    if (dtype == core::DataType::FLOAT32) {
        const float* a_data = static_cast<const float*>(a->data());
        const float* b_data = static_cast<const float*>(b->data());
        float* out_data = static_cast<float*>(out->data());

        for (int64_t i = 0; i < total; ++i) {
            int64_t a_idx = 0, b_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                a_idx += indices[d] * a_strides[d];
                b_idx += indices[d] * b_strides[d];
            }
            out_data[i] = op(a_data[a_idx], b_data[b_idx]);

            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < out_shape[d]) break;
                indices[d] = 0;
            }
        }
    } else if (dtype == core::DataType::INT64) {
        const int64_t* a_data = static_cast<const int64_t*>(a->data());
        const int64_t* b_data = static_cast<const int64_t*>(b->data());
        int64_t* out_data = static_cast<int64_t*>(out->data());

        for (int64_t i = 0; i < total; ++i) {
            int64_t a_idx = 0, b_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                a_idx += indices[d] * a_strides[d];
                b_idx += indices[d] * b_strides[d];
            }
            out_data[i] = static_cast<int64_t>(op(static_cast<float>(a_data[a_idx]),
                                                   static_cast<float>(b_data[b_idx])));

            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < out_shape[d]) break;
                indices[d] = 0;
            }
        }
    } else if (dtype == core::DataType::INT32) {
        const int32_t* a_data = static_cast<const int32_t*>(a->data());
        const int32_t* b_data = static_cast<const int32_t*>(b->data());
        int32_t* out_data = static_cast<int32_t*>(out->data());

        for (int64_t i = 0; i < total; ++i) {
            int64_t a_idx = 0, b_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                a_idx += indices[d] * a_strides[d];
                b_idx += indices[d] * b_strides[d];
            }
            out_data[i] = static_cast<int32_t>(op(static_cast<float>(a_data[a_idx]),
                                                   static_cast<float>(b_data[b_idx])));

            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < out_shape[d]) break;
                indices[d] = 0;
            }
        }
    } else {
        return core::Status::ERROR_NOT_IMPLEMENTED;
    }

    return core::Status::SUCCESS;
}

}  // namespace

// =============================================================================
// Add CPU Plugin
// =============================================================================

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

        core::Shape out_shape;
        auto status = broadcast_shapes(input_shapes[0], input_shapes[1], out_shape);
        if (status != core::Status::SUCCESS) {
            return status;
        }

        output_shapes.clear();
        output_shapes.push_back(out_shape);
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;
        return elementwise_binary_op(inputs, outputs,
            [](float a, float b) { return a + b; });
    }
};

// =============================================================================
// Sub CPU Plugin
// =============================================================================

class SubCPUPlugin : public SimpleCPUPlugin<SubCPUPlugin> {
public:
    SubCPUPlugin() = default;
    ~SubCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Sub";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kSUB;
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

        core::Shape out_shape;
        auto status = broadcast_shapes(input_shapes[0], input_shapes[1], out_shape);
        if (status != core::Status::SUCCESS) {
            return status;
        }

        output_shapes.clear();
        output_shapes.push_back(out_shape);
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;
        return elementwise_binary_op(inputs, outputs,
            [](float a, float b) { return a - b; });
    }
};

// =============================================================================
// Mul CPU Plugin
// =============================================================================

class MulCPUPlugin : public SimpleCPUPlugin<MulCPUPlugin> {
public:
    MulCPUPlugin() = default;
    ~MulCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Mul";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kMUL;
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

        core::Shape out_shape;
        auto status = broadcast_shapes(input_shapes[0], input_shapes[1], out_shape);
        if (status != core::Status::SUCCESS) {
            return status;
        }

        output_shapes.clear();
        output_shapes.push_back(out_shape);
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;
        return elementwise_binary_op(inputs, outputs,
            [](float a, float b) { return a * b; });
    }
};

// =============================================================================
// Div CPU Plugin
// =============================================================================

class DivCPUPlugin : public SimpleCPUPlugin<DivCPUPlugin> {
public:
    DivCPUPlugin() = default;
    ~DivCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Div";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kDIV;
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

        core::Shape out_shape;
        auto status = broadcast_shapes(input_shapes[0], input_shapes[1], out_shape);
        if (status != core::Status::SUCCESS) {
            return status;
        }

        output_shapes.clear();
        output_shapes.push_back(out_shape);
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;
        return elementwise_binary_op(inputs, outputs,
            [](float a, float b) { return a / b; });
    }
};

// Register plugins
REGISTER_PLUGIN_SIMPLE(AddCPUPlugin, "Add", kADD, CPU)
REGISTER_PLUGIN_SIMPLE(SubCPUPlugin, "Sub", kSUB, CPU)
REGISTER_PLUGIN_SIMPLE(MulCPUPlugin, "Mul", kMUL, CPU)
REGISTER_PLUGIN_SIMPLE(DivCPUPlugin, "Div", kDIV, CPU)

}  // namespace operators
}  // namespace mini_infer
