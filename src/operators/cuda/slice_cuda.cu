#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <algorithm>
#include <string>
#include <vector>

namespace mini_infer {
namespace operators {

namespace {

__global__ void slice_kernel(
    const float* input,
    float* output,
    const int64_t* starts,
    const int64_t* steps,
    const int64_t* in_strides,
    const int64_t* out_strides,
    const int64_t* out_dims,
    int ndim,
    int64_t out_total
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_total) return;

    // Convert output linear index to coordinates and map to input
    int64_t remaining = idx;
    int64_t in_idx = 0;

    for (int d = 0; d < ndim; ++d) {
        int64_t out_coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        int64_t in_coord = starts[d] + out_coord * steps[d];
        in_idx += in_coord * in_strides[d];
    }

    output[idx] = input[in_idx];
}

}  // namespace

class SliceCUDAPlugin : public SimpleCUDAPlugin<SliceCUDAPlugin> {
public:
    SliceCUDAPlugin() = default;
    ~SliceCUDAPlugin() override = default;

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
        // Dynamic: 1 (static params) or up to 5 (data, starts, ends, axes, steps)
        return -1;
    }

    void set_param(std::shared_ptr<PluginParam> param) override {
        param_ = std::dynamic_pointer_cast<SliceParam>(param);
    }

    const void* get_param_ptr() const noexcept override {
        return param_.get();
    }

    // Check if we have static parameters
    bool has_static_params() const {
        return param_ && !param_->starts.empty() && !param_->ends.empty();
    }

    // Helper to read int64 data from tensor
    static std::vector<int64_t> read_int64_tensor(const std::shared_ptr<core::Tensor>& tensor) {
        std::vector<int64_t> result;
        if (!tensor || !tensor->data()) return result;

        int64_t numel = tensor->shape().numel();
        const int64_t* data = static_cast<const int64_t*>(tensor->data());
        result.assign(data, data + numel);
        return result;
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {
        if (input_shapes.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_dims = input_shapes[0].dims();
        int64_t ndim = static_cast<int64_t>(input_dims.size());

        // Check if we have static params
        if (!has_static_params()) {
            // Dynamic slice - we cannot infer shape without tensor values
            // Return input shape as placeholder (will be corrected at runtime)
            output_shapes.clear();
            output_shapes.push_back(input_shapes[0]);
            return core::Status::SUCCESS;
        }

        std::vector<int64_t> axes = param_->axes;
        if (axes.empty()) {
            for (int64_t i = 0; i < static_cast<int64_t>(param_->starts.size()); ++i) {
                axes.push_back(i);
            }
        }
        for (auto& ax : axes) {
            if (ax < 0) ax += ndim;
        }

        std::vector<int64_t> steps = param_->steps;
        if (steps.empty()) {
            steps.resize(axes.size(), 1);
        }

        std::vector<int64_t> out_dims = input_dims;

        for (size_t i = 0; i < axes.size(); ++i) {
            int64_t axis = axes[i];
            if (axis < 0 || axis >= ndim) continue;

            if (i >= param_->starts.size() || i >= param_->ends.size()) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }

            int64_t dim_size = input_dims[axis];
            int64_t start = param_->starts[i];
            int64_t end = param_->ends[i];
            int64_t step = steps[i];

            if (start < 0) start += dim_size;
            start = std::max(int64_t(0), std::min(start, dim_size));

            if (end < 0) end += dim_size;
            if (end > dim_size) end = dim_size;
            end = std::max(int64_t(0), std::min(end, dim_size));

            int64_t out_dim = 0;
            if (step > 0) {
                out_dim = (end - start + step - 1) / step;
            } else if (step < 0) {
                out_dim = (start - end - step - 1) / (-step);
            }
            out_dim = std::max(int64_t(0), out_dim);

            out_dims[axis] = out_dim;
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(out_dims));
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {

        if (inputs.empty() || outputs.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input = inputs[0];
        auto& output = outputs[0];

        if (!input || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        if (!context.device_context) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(
            context.device_context
        );
        if (!cuda_ctx) {
            return core::Status::ERROR_BACKEND;
        }

        if (input->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& input_shape = input->shape();
        const auto& input_dims = input_shape.dims();
        const auto& output_shape = output->shape();
        const auto& output_dims = output_shape.dims();
        int ndim = static_cast<int>(input_dims.size());

        // Get slice parameters - either from static param or from input tensors
        std::vector<int64_t> starts_vec, ends_vec, axes_vec, steps_vec;

        if (has_static_params()) {
            // Use static parameters
            starts_vec = param_->starts;
            ends_vec = param_->ends;
            axes_vec = param_->axes;
            steps_vec = param_->steps;
        } else {
            // Read from input tensors (dynamic slice)
            // inputs[0] = data, inputs[1] = starts, inputs[2] = ends,
            // inputs[3] = axes (optional), inputs[4] = steps (optional)
            if (inputs.size() < 3) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }

            if (inputs[1]) starts_vec = read_int64_tensor(inputs[1]);
            if (inputs[2]) ends_vec = read_int64_tensor(inputs[2]);
            if (inputs.size() > 3 && inputs[3]) axes_vec = read_int64_tensor(inputs[3]);
            if (inputs.size() > 4 && inputs[4]) steps_vec = read_int64_tensor(inputs[4]);

            if (starts_vec.empty() || ends_vec.empty()) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
        }

        // Normalize axes
        if (axes_vec.empty()) {
            for (int64_t i = 0; i < static_cast<int64_t>(starts_vec.size()); ++i) {
                axes_vec.push_back(i);
            }
        }
        for (auto& ax : axes_vec) {
            if (ax < 0) ax += ndim;
        }

        // Normalize steps
        if (steps_vec.empty()) {
            steps_vec.resize(axes_vec.size(), 1);
        }

        // Build per-axis slice info
        std::vector<int64_t> starts(ndim, 0);
        std::vector<int64_t> slice_steps(ndim, 1);

        for (size_t i = 0; i < axes_vec.size(); ++i) {
            int64_t axis = axes_vec[i];
            if (axis < 0 || axis >= ndim) continue;

            if (i >= starts_vec.size()) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }

            int64_t dim_size = input_dims[axis];
            int64_t start = starts_vec[i];
            int64_t step = steps_vec[i];

            if (start < 0) start += dim_size;
            start = std::max(int64_t(0), std::min(start, dim_size));

            starts[axis] = start;
            slice_steps[axis] = step;
        }

        // Compute strides
        std::vector<int64_t> in_strides(ndim);
        int64_t stride = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            in_strides[i] = stride;
            stride *= input_dims[i];
        }

        std::vector<int64_t> out_strides(ndim);
        stride = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            out_strides[i] = stride;
            stride *= output_dims[i];
        }

        // Allocate device memory for slice parameters
        int64_t* d_starts;
        int64_t* d_steps;
        int64_t* d_in_strides;
        int64_t* d_out_strides;
        int64_t* d_out_dims;

        cudaMalloc(&d_starts, ndim * sizeof(int64_t));
        cudaMalloc(&d_steps, ndim * sizeof(int64_t));
        cudaMalloc(&d_in_strides, ndim * sizeof(int64_t));
        cudaMalloc(&d_out_strides, ndim * sizeof(int64_t));
        cudaMalloc(&d_out_dims, ndim * sizeof(int64_t));

        cudaMemcpyAsync(d_starts, starts.data(), ndim * sizeof(int64_t),
                       cudaMemcpyHostToDevice, cuda_ctx->stream());
        cudaMemcpyAsync(d_steps, slice_steps.data(), ndim * sizeof(int64_t),
                       cudaMemcpyHostToDevice, cuda_ctx->stream());
        cudaMemcpyAsync(d_in_strides, in_strides.data(), ndim * sizeof(int64_t),
                       cudaMemcpyHostToDevice, cuda_ctx->stream());
        cudaMemcpyAsync(d_out_strides, out_strides.data(), ndim * sizeof(int64_t),
                       cudaMemcpyHostToDevice, cuda_ctx->stream());
        cudaMemcpyAsync(d_out_dims, output_dims.data(), ndim * sizeof(int64_t),
                       cudaMemcpyHostToDevice, cuda_ctx->stream());

        const float* data_in = static_cast<const float*>(input->data());
        float* data_out = static_cast<float*>(output->data());
        int64_t out_total = output_shape.numel();

        const int threads = 256;
        int blocks = (out_total + threads - 1) / threads;

        slice_kernel<<<blocks, threads, 0, cuda_ctx->stream()>>>(
            data_in, data_out, d_starts, d_steps, d_in_strides, d_out_strides,
            d_out_dims, ndim, out_total
        );

        // Free device memory
        cudaFree(d_starts);
        cudaFree(d_steps);
        cudaFree(d_in_strides);
        cudaFree(d_out_strides);
        cudaFree(d_out_dims);

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }

private:
    std::shared_ptr<SliceParam> param_;
};

REGISTER_PLUGIN_SIMPLE(SliceCUDAPlugin, "Slice", kSLICE, CUDA)

}  // namespace operators
}  // namespace mini_infer
