#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace mini_infer {
namespace operators {

namespace {

/**
 * @brief Normalize slice parameters according to ONNX/NumPy semantics
 *
 * For positive step:
 *   - start defaults to 0, end defaults to dim_size
 *   - negative indices wrap around: idx -> idx + dim_size
 *   - start clamped to [0, dim_size], end clamped to [0, dim_size]
 *
 * For negative step:
 *   - start defaults to dim_size-1, end defaults to -1 (before first element)
 *   - negative indices wrap around: idx -> idx + dim_size
 *   - start clamped to [-1, dim_size-1], end clamped to [-1, dim_size-1]
 *   - end=-1 means "stop before index 0" (i.e., include index 0)
 */
void normalize_slice_params(int64_t dim_size, int64_t& start, int64_t& end, int64_t step) {
    constexpr int64_t INT_MAX_VAL = std::numeric_limits<int64_t>::max();
    constexpr int64_t INT_MIN_VAL = std::numeric_limits<int64_t>::min();

    if (step > 0) {
        // Handle INT_MAX for unbounded end
        if (end >= INT_MAX_VAL / 2) {
            end = dim_size;
        }
        // Handle INT_MIN for unbounded start
        if (start <= INT_MIN_VAL / 2) {
            start = 0;
        }

        // Normalize negative indices
        if (start < 0) {
            start += dim_size;
        }
        if (end < 0) {
            end += dim_size;
        }

        // Clamp to valid range [0, dim_size]
        start = std::max(int64_t(0), std::min(start, dim_size));
        end = std::max(int64_t(0), std::min(end, dim_size));
    } else {
        // step < 0: reverse slicing
        // Handle INT_MAX for unbounded start (means start from last element)
        if (start >= INT_MAX_VAL / 2) {
            start = dim_size - 1;
        }
        // Handle INT_MIN for unbounded end (means go to before first element)
        if (end <= INT_MIN_VAL / 2) {
            end = -1;  // -1 means "before index 0"
        }

        // Normalize negative indices
        // For negative step, negative index -1 after normalization should stay as dim_size-1
        // But end=-1 in the original input means "before index 0", not "last element"
        // So we need to be careful: only normalize if the value is in range [-dim_size, -1]
        if (start < 0 && start >= -dim_size) {
            start += dim_size;
        }
        // For end: -1 means "before index 0" (exclusive boundary)
        // Only normalize if it's a valid negative index referring to an element
        if (end < -1 && end >= -dim_size) {
            end += dim_size;
        }
        // end == -1 stays as -1 (meaning "stop before index 0")

        // Clamp start to valid range [-1, dim_size-1]
        // start can be dim_size-1 at most (last element)
        // start can be -1 at minimum (but that would give empty result)
        start = std::max(int64_t(-1), std::min(start, dim_size - 1));

        // Clamp end to valid range [-1, dim_size-1]
        // end=-1 means stop before index 0 (include index 0)
        // end=dim_size-1 means stop before last element (exclude last)
        end = std::max(int64_t(-1), std::min(end, dim_size - 1));
    }
}

/**
 * @brief Compute output dimension for a slice
 */
int64_t compute_slice_dim(int64_t start, int64_t end, int64_t step) {
    if (step > 0) {
        if (end <= start) {
            return 0;
        }
        return (end - start + step - 1) / step;
    } else {
        // step < 0
        // For reverse slice: start=3, end=-1, step=-1 should give 4 elements [3,2,1,0]
        // start - end = 3 - (-1) = 4, then divide by |step|
        if (start <= end) {
            return 0;
        }
        return (start - end + (-step) - 1) / (-step);
    }
}

}  // namespace

/**
 * @brief Slice CPU Plugin
 *
 * Extracts a slice from a tensor.
 * Supports both:
 * - Opset < 10: slice params from attributes (via param_)
 * - Opset >= 10: slice params from tensor inputs (data, starts, ends, [axes], [steps])
 */
class SliceCPUPlugin : public CPUPlugin<SliceCPUPlugin, SliceParam> {
public:
    SliceCPUPlugin() {
        param_ = std::make_shared<SliceParam>();
    }
    ~SliceCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Slice";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kSLICE;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        // Variable: 1 (data only, params from attributes) or 3-5 (opset >= 10)
        return -1;
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {

        if (input_shapes.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_shape = input_shapes[0];
        const int64_t ndim = static_cast<int64_t>(input_shape.ndim());

        // Use param_ for shape inference (populated by importer or runtime)
        if (!param_ || param_->starts.empty()) {
            // No slice params available, return input shape as-is
            // (will be resolved at runtime for dynamic slicing)
            output_shapes.clear();
            output_shapes.push_back(input_shape);
            return core::Status::SUCCESS;
        }

        std::vector<int64_t> output_dims = input_shape.dims();

        // Process each axis
        for (size_t i = 0; i < param_->starts.size(); ++i) {
            int64_t axis = (i < param_->axes.size()) ? param_->axes[i] : static_cast<int64_t>(i);
            if (axis < 0) {
                axis += ndim;
            }
            if (axis < 0 || axis >= ndim) {
                continue;  // Skip invalid axes
            }

            int64_t dim_size = input_shape[axis];
            int64_t start = param_->starts[i];
            int64_t end = (i < param_->ends.size()) ? param_->ends[i] : dim_size;
            int64_t step = (i < param_->steps.size()) ? param_->steps[i] : 1;

            if (step == 0) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }

            normalize_slice_params(dim_size, start, end, step);
            output_dims[axis] = compute_slice_dim(start, end, step);
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(output_dims));
        return core::Status::SUCCESS;
    }

    /**
     * @brief Infer output shapes with access to constant input tensors
     *
     * For ONNX opset >= 10, slice parameters come from input tensors.
     * This method reads starts/ends/axes/steps from constant input tensors.
     */
    core::Status infer_output_shapes_with_tensors(
        const std::vector<core::Shape>& input_shapes,
        const std::vector<core::DataType>& input_dtypes,
        const std::vector<std::shared_ptr<core::Tensor>>& input_tensors,
        std::vector<core::Shape>& output_shapes,
        std::vector<core::DataType>& output_dtypes) const override {

        if (input_shapes.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_shape = input_shapes[0];
        const int64_t ndim = static_cast<int64_t>(input_shape.ndim());

        // Helper to read int64 values from tensor
        auto read_int64_tensor = [](const std::shared_ptr<core::Tensor>& t, std::vector<int64_t>& vec) -> bool {
            if (!t || !t->data() || t->shape().numel() == 0) return false;
            const int64_t n = t->shape().numel();
            vec.resize(n);
            if (t->dtype() == core::DataType::INT64) {
                const int64_t* data = static_cast<const int64_t*>(t->data());
                for (int64_t i = 0; i < n; ++i) vec[i] = data[i];
                return true;
            } else if (t->dtype() == core::DataType::INT32) {
                const int32_t* data = static_cast<const int32_t*>(t->data());
                for (int64_t i = 0; i < n; ++i) vec[i] = static_cast<int64_t>(data[i]);
                return true;
            }
            return false;
        };

        std::vector<int64_t> starts_vec, ends_vec, axes_vec, steps_vec;

        // Try to read from input tensors (opset >= 10)
        if (input_tensors.size() >= 3) {
            read_int64_tensor(input_tensors.size() > 1 ? input_tensors[1] : nullptr, starts_vec);
            read_int64_tensor(input_tensors.size() > 2 ? input_tensors[2] : nullptr, ends_vec);
            if (input_tensors.size() > 3) {
                read_int64_tensor(input_tensors[3], axes_vec);
            }
            if (input_tensors.size() > 4) {
                read_int64_tensor(input_tensors[4], steps_vec);
            }
        }

        // Fall back to param_ if tensors not available
        if (starts_vec.empty() && param_ && !param_->starts.empty()) {
            starts_vec = param_->starts;
            ends_vec = param_->ends;
            axes_vec = param_->axes;
            steps_vec = param_->steps;
        }

        // If still no params, return input shape
        if (starts_vec.empty()) {
            output_shapes.clear();
            output_shapes.push_back(input_shape);
            output_dtypes.clear();
            output_dtypes.push_back(input_dtypes.empty() ? core::DataType::FLOAT32 : input_dtypes[0]);
            return core::Status::SUCCESS;
        }

        // Default axes if not provided
        if (axes_vec.empty()) {
            axes_vec.resize(starts_vec.size());
            for (size_t i = 0; i < starts_vec.size(); ++i) {
                axes_vec[i] = static_cast<int64_t>(i);
            }
        }

        // Default steps if not provided
        if (steps_vec.empty()) {
            steps_vec.resize(starts_vec.size(), 1);
        }

        std::vector<int64_t> output_dims = input_shape.dims();

        // Process each axis
        for (size_t i = 0; i < starts_vec.size(); ++i) {
            int64_t axis = axes_vec[i];
            if (axis < 0) {
                axis += ndim;
            }
            if (axis < 0 || axis >= ndim) {
                continue;
            }

            int64_t dim_size = input_shape[axis];
            int64_t start = starts_vec[i];
            int64_t end = (i < ends_vec.size()) ? ends_vec[i] : dim_size;
            int64_t step = (i < steps_vec.size()) ? steps_vec[i] : 1;

            if (step == 0) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }

            normalize_slice_params(dim_size, start, end, step);
            output_dims[axis] = compute_slice_dim(start, end, step);
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(output_dims));
        output_dtypes.clear();
        output_dtypes.push_back(input_dtypes.empty() ? core::DataType::FLOAT32 : input_dtypes[0]);
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;

        // Support 1 input (opset < 10) or 3-5 inputs (opset >= 10)
        if (inputs.empty() || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input = inputs[0];
        auto& output = outputs[0];

        if (!input || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const core::DataType dtype = input->dtype();
        if (dtype != core::DataType::FLOAT32 && dtype != core::DataType::INT64 && dtype != core::DataType::INT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const core::Shape& in_shape = input->shape();
        const core::Shape& out_shape = output->shape();
        const int64_t ndim = static_cast<int64_t>(in_shape.ndim());

        // Extract slice parameters
        std::vector<int64_t> starts_vec, ends_vec, axes_vec, steps_vec;

        if (inputs.size() >= 3) {
            // Opset >= 10: read from tensor inputs
            // inputs[1] = starts, inputs[2] = ends, inputs[3] = axes (optional), inputs[4] = steps (optional)
            auto read_int64_tensor = [](const std::shared_ptr<core::Tensor>& t, std::vector<int64_t>& vec) {
                if (!t) return;
                const int64_t n = t->shape().numel();
                vec.resize(n);
                if (t->dtype() == core::DataType::INT64) {
                    const int64_t* data = static_cast<const int64_t*>(t->data());
                    for (int64_t i = 0; i < n; ++i) vec[i] = data[i];
                } else if (t->dtype() == core::DataType::INT32) {
                    const int32_t* data = static_cast<const int32_t*>(t->data());
                    for (int64_t i = 0; i < n; ++i) vec[i] = static_cast<int64_t>(data[i]);
                }
            };

            read_int64_tensor(inputs[1], starts_vec);
            read_int64_tensor(inputs[2], ends_vec);
            if (inputs.size() >= 4 && inputs[3]) {
                read_int64_tensor(inputs[3], axes_vec);
            }
            if (inputs.size() >= 5 && inputs[4]) {
                read_int64_tensor(inputs[4], steps_vec);
            }
        } else if (param_) {
            // Opset < 10: read from attributes
            starts_vec = param_->starts;
            ends_vec = param_->ends;
            axes_vec = param_->axes;
            steps_vec = param_->steps;
        }

        if (starts_vec.empty()) {
            // No slicing, just copy
            std::memcpy(output->data(), input->data(), input->size_in_bytes());
            return core::Status::SUCCESS;
        }

        // Default axes if not provided
        if (axes_vec.empty()) {
            axes_vec.resize(starts_vec.size());
            for (size_t i = 0; i < starts_vec.size(); ++i) {
                axes_vec[i] = static_cast<int64_t>(i);
            }
        }

        // Default steps if not provided
        if (steps_vec.empty()) {
            steps_vec.resize(starts_vec.size(), 1);
        }

        // Build slice parameters for all axes
        // For positive step: start from 0, end at dim_size
        // For negative step: start from dim_size-1, end at -1
        std::vector<int64_t> starts(ndim);
        std::vector<int64_t> ends(ndim);
        std::vector<int64_t> steps(ndim, 1);

        for (int64_t d = 0; d < ndim; ++d) {
            starts[d] = 0;
            ends[d] = in_shape[d];
        }

        for (size_t i = 0; i < starts_vec.size(); ++i) {
            int64_t axis = axes_vec[i];
            if (axis < 0) axis += ndim;
            if (axis < 0 || axis >= ndim) continue;

            int64_t dim_size = in_shape[axis];
            int64_t start = starts_vec[i];
            int64_t end = ends_vec[i];
            int64_t step = (i < steps_vec.size()) ? steps_vec[i] : 1;

            if (step == 0) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }

            normalize_slice_params(dim_size, start, end, step);

            starts[axis] = start;
            ends[axis] = end;
            steps[axis] = step;
        }

        // Compute input strides
        std::vector<int64_t> in_strides(ndim);
        int64_t stride = 1;
        for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
            in_strides[i] = stride;
            stride *= in_shape[i];
        }

        const int64_t total = out_shape.numel();
        std::vector<int64_t> out_indices(ndim, 0);

        // Use template-like approach for different dtypes
        if (dtype == core::DataType::FLOAT32) {
            const float* in_data = static_cast<const float*>(input->data());
            float* out_data = static_cast<float*>(output->data());

            for (int64_t i = 0; i < total; ++i) {
                int64_t in_idx = 0;
                for (int64_t d = 0; d < ndim; ++d) {
                    int64_t in_coord = starts[d] + out_indices[d] * steps[d];
                    in_idx += in_coord * in_strides[d];
                }
                out_data[i] = in_data[in_idx];

                for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                    out_indices[d]++;
                    if (out_indices[d] < out_shape[d]) break;
                    out_indices[d] = 0;
                }
            }
        } else if (dtype == core::DataType::INT64) {
            const int64_t* in_data = static_cast<const int64_t*>(input->data());
            int64_t* out_data = static_cast<int64_t*>(output->data());

            for (int64_t i = 0; i < total; ++i) {
                int64_t in_idx = 0;
                for (int64_t d = 0; d < ndim; ++d) {
                    int64_t in_coord = starts[d] + out_indices[d] * steps[d];
                    in_idx += in_coord * in_strides[d];
                }
                out_data[i] = in_data[in_idx];

                for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                    out_indices[d]++;
                    if (out_indices[d] < out_shape[d]) break;
                    out_indices[d] = 0;
                }
            }
        } else if (dtype == core::DataType::INT32) {
            const int32_t* in_data = static_cast<const int32_t*>(input->data());
            int32_t* out_data = static_cast<int32_t*>(output->data());

            for (int64_t i = 0; i < total; ++i) {
                int64_t in_idx = 0;
                for (int64_t d = 0; d < ndim; ++d) {
                    int64_t in_coord = starts[d] + out_indices[d] * steps[d];
                    in_idx += in_coord * in_strides[d];
                }
                out_data[i] = in_data[in_idx];

                for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                    out_indices[d]++;
                    if (out_indices[d] < out_shape[d]) break;
                    out_indices[d] = 0;
                }
            }
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(SliceCPUPlugin, "Slice", kSLICE, CPU)

}  // namespace operators
}  // namespace mini_infer
