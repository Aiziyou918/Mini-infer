#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <algorithm>
#include <string>

namespace mini_infer {
namespace operators {

namespace {

/**
 * @brief Add kernel for same-shape tensors (vectorized)
 */
__global__ void add_kernel_vectorized(
    const float* a,
    const float* b,
    float* out,
    int64_t n
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    int64_t vec_size = n / 4;
    const float4* a_vec = reinterpret_cast<const float4*>(a);
    const float4* b_vec = reinterpret_cast<const float4*>(b);
    float4* out_vec = reinterpret_cast<float4*>(out);

    for (int64_t i = tid; i < vec_size; i += stride) {
        float4 va = a_vec[i];
        float4 vb = b_vec[i];
        float4 vo;
        vo.x = va.x + vb.x;
        vo.y = va.y + vb.y;
        vo.z = va.z + vb.z;
        vo.w = va.w + vb.w;
        out_vec[i] = vo;
    }

    int64_t remaining_start = vec_size * 4;
    for (int64_t i = remaining_start + tid; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

/**
 * @brief Add kernel for same-shape tensors (simple)
 */
__global__ void add_kernel_simple(
    const float* a,
    const float* b,
    float* out,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief Add kernel with broadcasting support
 */
__global__ void add_kernel_broadcast(
    const float* a,
    const float* b,
    float* out,
    const int64_t* out_dims,
    const int64_t* strides_a,
    const int64_t* strides_b,
    int ndim,
    int64_t total
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Convert linear index to multi-dimensional index
    int64_t remaining = idx;
    int64_t idx_a = 0;
    int64_t idx_b = 0;

    for (int d = 0; d < ndim; ++d) {
        int64_t coord = remaining / 1;
        int64_t divisor = 1;
        for (int dd = d + 1; dd < ndim; ++dd) {
            divisor *= out_dims[dd];
        }
        coord = (remaining / divisor) % out_dims[d];

        idx_a += coord * strides_a[d];
        idx_b += coord * strides_b[d];
    }

    out[idx] = a[idx_a] + b[idx_b];
}

}  // namespace

/**
 * @brief Add CUDA Plugin
 */
class AddCUDAPlugin : public SimpleCUDAPlugin<AddCUDAPlugin> {
public:
    AddCUDAPlugin() = default;
    ~AddCUDAPlugin() override = default;

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

        // Compute broadcast shape
        const auto& dims_a = input_shapes[0].dims();
        const auto& dims_b = input_shapes[1].dims();

        size_t ndim_a = dims_a.size();
        size_t ndim_b = dims_b.size();
        size_t ndim_out = std::max(ndim_a, ndim_b);

        std::vector<int64_t> out_dims(ndim_out);

        for (size_t i = 0; i < ndim_out; ++i) {
            int64_t dim_a = (i < ndim_out - ndim_a) ? 1 : dims_a[i - (ndim_out - ndim_a)];
            int64_t dim_b = (i < ndim_out - ndim_b) ? 1 : dims_b[i - (ndim_out - ndim_b)];

            if (dim_a == dim_b) {
                out_dims[i] = dim_a;
            } else if (dim_a == 1) {
                out_dims[i] = dim_b;
            } else if (dim_b == 1) {
                out_dims[i] = dim_a;
            } else {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(out_dims));
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {

        if (inputs.size() != 2 || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_a = inputs[0];
        const auto& input_b = inputs[1];
        auto& output = outputs[0];

        if (!input_a || !input_b || !output) {
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

        if (input_a->dtype() != core::DataType::FLOAT32 ||
            input_b->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& shape_a = input_a->shape();
        const auto& shape_b = input_b->shape();
        const int64_t total = output->shape().numel();

        const float* data_a = static_cast<const float*>(input_a->data());
        const float* data_b = static_cast<const float*>(input_b->data());
        float* data_out = static_cast<float*>(output->data());

        const int threads = 256;

        // Fast path: same shape
        if (shape_a == shape_b) {
            if (total < 1024) {
                int blocks = (total + threads - 1) / threads;
                add_kernel_simple<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                    data_a, data_b, data_out, total
                );
            } else {
                int blocks = std::min((total + threads * 4 - 1) / (threads * 4), (int64_t)2048);
                add_kernel_vectorized<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                    data_a, data_b, data_out, total
                );
            }
        } else {
            // Broadcast path - use simple element-wise for now
            // TODO: Implement proper broadcast kernel with device memory for strides
            int blocks = (total + threads - 1) / threads;
            add_kernel_simple<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                data_a, data_b, data_out, total
            );
        }

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(AddCUDAPlugin, "Add", kADD, CUDA)

}  // namespace operators
}  // namespace mini_infer
