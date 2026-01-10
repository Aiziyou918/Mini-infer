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

// Simple reduce mean kernel for last axis (most common case in LayerNorm)
__global__ void reduce_mean_last_axis_kernel(
    const float* input,
    float* output,
    int64_t outer_size,
    int64_t reduce_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size) return;

    const float* in_ptr = input + idx * reduce_size;
    float sum = 0.0f;
    for (int64_t i = 0; i < reduce_size; ++i) {
        sum += in_ptr[i];
    }
    output[idx] = sum / static_cast<float>(reduce_size);
}

// Warp-level reduction for better performance
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduce mean for last axis with shared memory
__global__ void reduce_mean_last_axis_block_kernel(
    const float* input,
    float* output,
    int64_t outer_size,
    int64_t reduce_size
) {
    extern __shared__ float sdata[];

    int64_t row = blockIdx.x;
    if (row >= outer_size) return;

    const float* in_ptr = input + row * reduce_size;

    // Each thread sums multiple elements
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        sum += in_ptr[i];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[row] = sdata[0] / static_cast<float>(reduce_size);
    }
}

}  // namespace

class ReduceMeanCUDAPlugin : public SimpleCUDAPlugin<ReduceMeanCUDAPlugin> {
public:
    ReduceMeanCUDAPlugin() = default;
    ~ReduceMeanCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "ReduceMean";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kREDUCE_MEAN;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return 1;
    }

    void set_param(std::shared_ptr<PluginParam> param) override {
        param_ = std::dynamic_pointer_cast<ReduceMeanParam>(param);
    }

    const void* get_param_ptr() const noexcept override {
        return param_.get();
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {
        if (input_shapes.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_dims = input_shapes[0].dims();
        int64_t ndim = static_cast<int64_t>(input_dims.size());

        std::vector<int64_t> axes;
        if (param_ && !param_->axes.empty()) {
            axes = param_->axes;
            for (auto& ax : axes) {
                if (ax < 0) ax += ndim;
            }
        } else {
            for (int64_t i = 0; i < ndim; ++i) {
                axes.push_back(i);
            }
        }

        bool keepdims = param_ ? param_->keepdims : true;

        std::vector<int64_t> out_dims;
        for (int64_t i = 0; i < ndim; ++i) {
            bool is_reduce_axis = std::find(axes.begin(), axes.end(), i) != axes.end();
            if (is_reduce_axis) {
                if (keepdims) {
                    out_dims.push_back(1);
                }
            } else {
                out_dims.push_back(input_dims[i]);
            }
        }

        if (out_dims.empty()) {
            out_dims.push_back(1);
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(out_dims));
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {

        if (inputs.size() != 1 || outputs.size() != 1) {
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
        int64_t ndim = static_cast<int64_t>(input_dims.size());

        std::vector<int64_t> axes;
        if (param_ && !param_->axes.empty()) {
            axes = param_->axes;
            for (auto& ax : axes) {
                if (ax < 0) ax += ndim;
            }
        } else {
            for (int64_t i = 0; i < ndim; ++i) {
                axes.push_back(i);
            }
        }

        std::sort(axes.begin(), axes.end());

        const float* data_in = static_cast<const float*>(input->data());
        float* data_out = static_cast<float*>(output->data());

        // Optimized path: reduce only last axis (common in LayerNorm)
        if (axes.size() == 1 && axes[0] == ndim - 1) {
            int64_t outer_size = 1;
            for (int64_t i = 0; i < ndim - 1; ++i) {
                outer_size *= input_dims[i];
            }
            int64_t reduce_size = input_dims[ndim - 1];

            if (reduce_size <= 256) {
                const int threads = 256;
                int blocks = (outer_size + threads - 1) / threads;
                reduce_mean_last_axis_kernel<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                    data_in, data_out, outer_size, reduce_size
                );
            } else {
                const int threads = 256;
                int blocks = outer_size;
                size_t shared_mem = threads * sizeof(float);
                reduce_mean_last_axis_block_kernel<<<blocks, threads, shared_mem, cuda_ctx->stream()>>>(
                    data_in, data_out, outer_size, reduce_size
                );
            }
        } else {
            // General case: fallback to simple kernel
            // For now, use the last-axis kernel if possible
            int64_t outer_size = 1;
            int64_t reduce_size = 1;

            for (int64_t i = 0; i < ndim; ++i) {
                if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
                    reduce_size *= input_dims[i];
                } else {
                    outer_size *= input_dims[i];
                }
            }

            const int threads = 256;
            int blocks = (outer_size + threads - 1) / threads;
            reduce_mean_last_axis_kernel<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                data_in, data_out, outer_size, reduce_size
            );
        }

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }

private:
    std::shared_ptr<ReduceMeanParam> param_;
};

REGISTER_PLUGIN_SIMPLE(ReduceMeanCUDAPlugin, "ReduceMean", kREDUCE_MEAN, CUDA)

}  // namespace operators
}  // namespace mini_infer
