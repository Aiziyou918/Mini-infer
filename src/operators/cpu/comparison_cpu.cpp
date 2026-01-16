#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>
#include <cstring>

namespace mini_infer {
namespace operators {

// =============================================================================
// Helper: Broadcast shape computation
// =============================================================================

namespace {

bool compute_broadcast_shape(const core::Shape& a, const core::Shape& b,
                             std::vector<int64_t>& out_dims) {
    const size_t ndim_a = a.ndim();
    const size_t ndim_b = b.ndim();
    const size_t ndim_out = std::max(ndim_a, ndim_b);

    out_dims.resize(ndim_out);
    for (size_t i = 0; i < ndim_out; ++i) {
        const int64_t dim_a = (i < ndim_a) ? a[ndim_a - 1 - i] : 1;
        const int64_t dim_b = (i < ndim_b) ? b[ndim_b - 1 - i] : 1;

        if (dim_a == dim_b) {
            out_dims[ndim_out - 1 - i] = dim_a;
        } else if (dim_a == 1) {
            out_dims[ndim_out - 1 - i] = dim_b;
        } else if (dim_b == 1) {
            out_dims[ndim_out - 1 - i] = dim_a;
        } else {
            return false;  // Incompatible shapes
        }
    }
    return true;
}

// Compute broadcast strides for a tensor with shape `shape` to broadcast to `out_shape`
void compute_broadcast_strides(const core::Shape& shape, const core::Shape& out_shape,
                               std::vector<int64_t>& strides) {
    const size_t ndim = out_shape.ndim();
    const size_t shape_ndim = shape.ndim();
    strides.resize(ndim);

    int64_t stride = 1;
    for (int i = static_cast<int>(shape_ndim) - 1; i >= 0; --i) {
        if (shape[i] == 1) {
            strides[ndim - shape_ndim + i] = 0;  // Broadcast dimension
        } else {
            strides[ndim - shape_ndim + i] = stride;
            stride *= shape[i];
        }
    }
    // Fill leading dimensions with 0 (broadcast)
    for (size_t i = 0; i < ndim - shape_ndim; ++i) {
        strides[i] = 0;
    }
}

int64_t compute_index(const std::vector<int64_t>& indices,
                      const std::vector<int64_t>& strides) {
    int64_t idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        idx += indices[i] * strides[i];
    }
    return idx;
}

}  // namespace

// =============================================================================
// Equal CPU Plugin
// =============================================================================

class EqualCPUPlugin : public SimpleCPUPlugin<EqualCPUPlugin> {
public:
    EqualCPUPlugin() = default;
    ~EqualCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Equal";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kEQUAL;
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

        std::vector<int64_t> out_dims;
        if (!compute_broadcast_shape(input_shapes[0], input_shapes[1], out_dims)) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(out_dims));
        return core::Status::SUCCESS;
    }

    core::Status infer_output_metadata(
        const std::vector<core::Shape>& input_shapes,
        const std::vector<core::DataType>& input_dtypes,
        std::vector<core::Shape>& output_shapes,
        std::vector<core::DataType>& output_dtypes) const override {
        (void)input_dtypes;

        auto status = infer_output_shapes(input_shapes, output_shapes);
        if (status != core::Status::SUCCESS) {
            return status;
        }

        // Equal outputs BOOL (we use INT32 as bool representation)
        output_dtypes.clear();
        output_dtypes.push_back(core::DataType::INT32);
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

        const auto& a = inputs[0];
        const auto& b = inputs[1];
        auto& out = outputs[0];

        if (!a || !b || !out) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const core::Shape& out_shape = out->shape();
        const int64_t total = out_shape.numel();
        const size_t ndim = out_shape.ndim();

        std::vector<int64_t> strides_a, strides_b;
        compute_broadcast_strides(a->shape(), out_shape, strides_a);
        compute_broadcast_strides(b->shape(), out_shape, strides_b);

        int32_t* out_data = static_cast<int32_t*>(out->data());
        std::vector<int64_t> indices(ndim, 0);

        // Support FLOAT32 and INT64 comparisons
        if (a->dtype() == core::DataType::FLOAT32 && b->dtype() == core::DataType::FLOAT32) {
            const float* a_data = static_cast<const float*>(a->data());
            const float* b_data = static_cast<const float*>(b->data());

            for (int64_t i = 0; i < total; ++i) {
                int64_t idx_a = compute_index(indices, strides_a);
                int64_t idx_b = compute_index(indices, strides_b);
                out_data[i] = (a_data[idx_a] == b_data[idx_b]) ? 1 : 0;

                // Increment indices
                for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                    indices[d]++;
                    if (indices[d] < out_shape[d]) break;
                    indices[d] = 0;
                }
            }
        } else if (a->dtype() == core::DataType::INT64 && b->dtype() == core::DataType::INT64) {
            const int64_t* a_data = static_cast<const int64_t*>(a->data());
            const int64_t* b_data = static_cast<const int64_t*>(b->data());

            for (int64_t i = 0; i < total; ++i) {
                int64_t idx_a = compute_index(indices, strides_a);
                int64_t idx_b = compute_index(indices, strides_b);
                out_data[i] = (a_data[idx_a] == b_data[idx_b]) ? 1 : 0;

                for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                    indices[d]++;
                    if (indices[d] < out_shape[d]) break;
                    indices[d] = 0;
                }
            }
        } else if (a->dtype() == core::DataType::INT32 && b->dtype() == core::DataType::INT32) {
            const int32_t* a_data = static_cast<const int32_t*>(a->data());
            const int32_t* b_data = static_cast<const int32_t*>(b->data());

            for (int64_t i = 0; i < total; ++i) {
                int64_t idx_a = compute_index(indices, strides_a);
                int64_t idx_b = compute_index(indices, strides_b);
                out_data[i] = (a_data[idx_a] == b_data[idx_b]) ? 1 : 0;

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
};

// =============================================================================
// Where CPU Plugin
// =============================================================================

class WhereCPUPlugin : public SimpleCPUPlugin<WhereCPUPlugin> {
public:
    WhereCPUPlugin() = default;
    ~WhereCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Where";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kWHERE;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return 3;  // condition, x, y
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {

        if (input_shapes.size() != 3) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Broadcast all three inputs
        std::vector<int64_t> temp_dims, out_dims;
        if (!compute_broadcast_shape(input_shapes[0], input_shapes[1], temp_dims)) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
        if (!compute_broadcast_shape(core::Shape(temp_dims), input_shapes[2], out_dims)) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(out_dims));
        return core::Status::SUCCESS;
    }

    core::Status infer_output_metadata(
        const std::vector<core::Shape>& input_shapes,
        const std::vector<core::DataType>& input_dtypes,
        std::vector<core::Shape>& output_shapes,
        std::vector<core::DataType>& output_dtypes) const override {

        auto status = infer_output_shapes(input_shapes, output_shapes);
        if (status != core::Status::SUCCESS) {
            return status;
        }

        // Output dtype should match x/y dtype (input[1] or input[2]), not condition dtype
        output_dtypes.clear();
        if (input_dtypes.size() >= 2) {
            output_dtypes.push_back(input_dtypes[1]);  // Use x's dtype
        } else {
            output_dtypes.push_back(core::DataType::FLOAT32);
        }
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;

        if (inputs.size() != 3 || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& cond = inputs[0];
        const auto& x = inputs[1];
        const auto& y = inputs[2];
        auto& out = outputs[0];

        if (!cond || !x || !y || !out) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const core::Shape& out_shape = out->shape();
        const int64_t total = out_shape.numel();
        const size_t ndim = out_shape.ndim();

        std::vector<int64_t> strides_cond, strides_x, strides_y;
        compute_broadcast_strides(cond->shape(), out_shape, strides_cond);
        compute_broadcast_strides(x->shape(), out_shape, strides_x);
        compute_broadcast_strides(y->shape(), out_shape, strides_y);

        std::vector<int64_t> indices(ndim, 0);

        // Condition can be INT32 (bool) or BOOL
        auto get_cond_value = [&](int64_t idx) -> bool {
            if (cond->dtype() == core::DataType::INT32) {
                return static_cast<const int32_t*>(cond->data())[idx] != 0;
            } else if (cond->dtype() == core::DataType::INT64) {
                return static_cast<const int64_t*>(cond->data())[idx] != 0;
            } else if (cond->dtype() == core::DataType::FLOAT32) {
                return static_cast<const float*>(cond->data())[idx] != 0.0f;
            }
            return false;
        };

        if (x->dtype() == core::DataType::FLOAT32) {
            const float* x_data = static_cast<const float*>(x->data());
            const float* y_data = static_cast<const float*>(y->data());
            float* out_data = static_cast<float*>(out->data());

            for (int64_t i = 0; i < total; ++i) {
                int64_t idx_cond = compute_index(indices, strides_cond);
                int64_t idx_x = compute_index(indices, strides_x);
                int64_t idx_y = compute_index(indices, strides_y);

                out_data[i] = get_cond_value(idx_cond) ? x_data[idx_x] : y_data[idx_y];

                for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                    indices[d]++;
                    if (indices[d] < out_shape[d]) break;
                    indices[d] = 0;
                }
            }
        } else if (x->dtype() == core::DataType::INT64) {
            const int64_t* x_data = static_cast<const int64_t*>(x->data());
            const int64_t* y_data = static_cast<const int64_t*>(y->data());
            int64_t* out_data = static_cast<int64_t*>(out->data());

            for (int64_t i = 0; i < total; ++i) {
                int64_t idx_cond = compute_index(indices, strides_cond);
                int64_t idx_x = compute_index(indices, strides_x);
                int64_t idx_y = compute_index(indices, strides_y);

                out_data[i] = get_cond_value(idx_cond) ? x_data[idx_x] : y_data[idx_y];

                for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                    indices[d]++;
                    if (indices[d] < out_shape[d]) break;
                    indices[d] = 0;
                }
            }
        } else if (x->dtype() == core::DataType::INT32) {
            const int32_t* x_data = static_cast<const int32_t*>(x->data());
            const int32_t* y_data = static_cast<const int32_t*>(y->data());
            int32_t* out_data = static_cast<int32_t*>(out->data());

            for (int64_t i = 0; i < total; ++i) {
                int64_t idx_cond = compute_index(indices, strides_cond);
                int64_t idx_x = compute_index(indices, strides_x);
                int64_t idx_y = compute_index(indices, strides_y);

                out_data[i] = get_cond_value(idx_cond) ? x_data[idx_x] : y_data[idx_y];

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
};

// =============================================================================
// Expand CPU Plugin
// =============================================================================

class ExpandCPUPlugin : public CPUPlugin<ExpandCPUPlugin, ExpandParam> {
public:
    ExpandCPUPlugin() {
        param_ = std::make_shared<ExpandParam>();
    }
    ~ExpandCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Expand";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kEXPAND;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return 2;  // input, shape
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {

        if (input_shapes.size() != 2) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        output_shapes.clear();

        if (param_ && !param_->shape.empty()) {
            auto status = validate_target_shape(input_shapes[0], param_->shape);
            if (status != core::Status::SUCCESS) {
                return status;
            }

            output_shapes.emplace_back(core::Shape(param_->shape));
        } else {
            // Fallback placeholder when target shape is not statically known.
            output_shapes.push_back(input_shapes[0]);
        }

        return core::Status::SUCCESS;
    }

    core::Status infer_output_shapes_with_tensors(
        const std::vector<core::Shape>& input_shapes,
        const std::vector<core::DataType>& input_dtypes,
        const std::vector<std::shared_ptr<core::Tensor>>& input_tensors,
        std::vector<core::Shape>& output_shapes,
        std::vector<core::DataType>& output_dtypes) const override {

        if (input_shapes.size() != 2) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        output_shapes.clear();
        output_dtypes.clear();

        // Try to extract target shape from shape tensor (input[1])
        std::vector<int64_t> target_shape;
        if (input_tensors.size() > 1 && input_tensors[1] && input_tensors[1]->data()) {
            auto status = extract_target_shape(input_tensors[1], target_shape);
            if (status == core::Status::SUCCESS) {
                status = validate_target_shape(input_shapes[0], target_shape);
                if (status == core::Status::SUCCESS) {
                    output_shapes.emplace_back(core::Shape(target_shape));
                    output_dtypes.push_back(input_dtypes.empty() ? core::DataType::FLOAT32 : input_dtypes[0]);
                    return core::Status::SUCCESS;
                }
            }
        }

        // Fallback to param or input shape
        if (param_ && !param_->shape.empty()) {
            auto status = validate_target_shape(input_shapes[0], param_->shape);
            if (status != core::Status::SUCCESS) {
                return status;
            }
            output_shapes.emplace_back(core::Shape(param_->shape));
        } else {
            output_shapes.push_back(input_shapes[0]);
        }

        output_dtypes.push_back(input_dtypes.empty() ? core::DataType::FLOAT32 : input_dtypes[0]);
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

        const auto& input = inputs[0];
        const auto& shape_tensor = inputs[1];
        auto& output = outputs[0];

        if (!input || !shape_tensor || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        std::vector<int64_t> target_shape;
        auto status = extract_target_shape(shape_tensor, target_shape);
        if (status != core::Status::SUCCESS) {
            if (param_ && !param_->shape.empty()) {
                target_shape = param_->shape;
            } else {
                return status;
            }
        }

        status = validate_target_shape(input->shape(), target_shape);
        if (status != core::Status::SUCCESS) {
            return status;
        }

        core::Shape desired_shape(target_shape);
        if (output->shape() != desired_shape) {
            output->resize(desired_shape);
        }

        const core::Shape& in_shape = input->shape();
        const core::Shape& out_shape = output->shape();
        const size_t ndim = out_shape.ndim();

        // Compute broadcast strides
        std::vector<int64_t> in_strides;
        compute_broadcast_strides(in_shape, out_shape, in_strides);

        const int64_t total = out_shape.numel();
        std::vector<int64_t> indices(ndim, 0);

        if (input->dtype() == core::DataType::FLOAT32) {
            const float* in_data = static_cast<const float*>(input->data());
            float* out_data = static_cast<float*>(output->data());

            for (int64_t i = 0; i < total; ++i) {
                int64_t in_idx = compute_index(indices, in_strides);
                out_data[i] = in_data[in_idx];

                for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                    indices[d]++;
                    if (indices[d] < out_shape[d]) break;
                    indices[d] = 0;
                }
            }
        } else if (input->dtype() == core::DataType::INT64) {
            const int64_t* in_data = static_cast<const int64_t*>(input->data());
            int64_t* out_data = static_cast<int64_t*>(output->data());

            for (int64_t i = 0; i < total; ++i) {
                int64_t in_idx = compute_index(indices, in_strides);
                out_data[i] = in_data[in_idx];

                for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                    indices[d]++;
                    if (indices[d] < out_shape[d]) break;
                    indices[d] = 0;
                }
            }
        } else if (input->dtype() == core::DataType::INT32) {
            const int32_t* in_data = static_cast<const int32_t*>(input->data());
            int32_t* out_data = static_cast<int32_t*>(output->data());

            for (int64_t i = 0; i < total; ++i) {
                int64_t in_idx = compute_index(indices, in_strides);
                out_data[i] = in_data[in_idx];

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

private:
    core::Status extract_target_shape(
        const std::shared_ptr<core::Tensor>& shape_tensor,
        std::vector<int64_t>& target_shape) const {
        target_shape.clear();

        if (!shape_tensor) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const size_t numel = static_cast<size_t>(shape_tensor->shape().numel());
        if (numel == 0) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const void* raw_data = shape_tensor->data();
        if (!raw_data) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        target_shape.resize(numel);
        if (shape_tensor->dtype() == core::DataType::INT64) {
            const int64_t* data = static_cast<const int64_t*>(raw_data);
            for (size_t i = 0; i < numel; ++i) {
                target_shape[i] = data[i];
            }
        } else if (shape_tensor->dtype() == core::DataType::INT32) {
            const int32_t* data = static_cast<const int32_t*>(raw_data);
            for (size_t i = 0; i < numel; ++i) {
                target_shape[i] = static_cast<int64_t>(data[i]);
            }
        } else {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        return core::Status::SUCCESS;
    }

    core::Status validate_target_shape(
        const core::Shape& input_shape,
        const std::vector<int64_t>& target_shape) const {
        if (target_shape.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const size_t in_rank = input_shape.ndim();
        const size_t out_rank = target_shape.size();
        if (out_rank < in_rank) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        for (size_t i = 0; i < out_rank; ++i) {
            const int64_t target_dim = target_shape[out_rank - 1 - i];
            if (target_dim < 0) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }

            const int64_t input_dim = (i < in_rank) ? input_shape[in_rank - 1 - i] : 1;
            if (input_dim != 1 && input_dim != target_dim) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
        }

        return core::Status::SUCCESS;
    }
};

// Register plugins
REGISTER_PLUGIN_SIMPLE(EqualCPUPlugin, "Equal", kEQUAL, CPU)
REGISTER_PLUGIN_SIMPLE(WhereCPUPlugin, "Where", kWHERE, CPU)
REGISTER_PLUGIN_SIMPLE(ExpandCPUPlugin, "Expand", kEXPAND, CPU)

}  // namespace operators
}  // namespace mini_infer
