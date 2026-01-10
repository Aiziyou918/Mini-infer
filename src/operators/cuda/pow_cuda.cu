#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <algorithm>
#include <string>

namespace mini_infer {
namespace operators {

namespace {

__global__ void pow_kernel_vectorized(
    const float* base,
    const float* exp,
    float* out,
    int64_t n
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        out[i] = powf(base[i], exp[i]);
    }
}

__global__ void pow_kernel_scalar_exp(
    const float* base,
    float exp_val,
    float* out,
    int64_t n
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        out[i] = powf(base[i], exp_val);
    }
}

__global__ void pow_kernel_simple(
    const float* base,
    const float* exp,
    float* out,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = powf(base[idx], exp[idx]);
    }
}

}  // namespace

class PowCUDAPlugin : public SimpleCUDAPlugin<PowCUDAPlugin> {
public:
    PowCUDAPlugin() = default;
    ~PowCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Pow";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kPOW;
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

        const auto& base = inputs[0];
        const auto& exponent = inputs[1];
        auto& output = outputs[0];

        if (!base || !exponent || !output) {
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

        if (base->dtype() != core::DataType::FLOAT32 ||
            exponent->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& shape_base = base->shape();
        const auto& shape_exp = exponent->shape();
        const int64_t total = output->shape().numel();

        const float* data_base = static_cast<const float*>(base->data());
        const float* data_exp = static_cast<const float*>(exponent->data());
        float* data_out = static_cast<float*>(output->data());

        const int threads = 256;
        int blocks = (total + threads - 1) / threads;

        // Fast path: scalar exponent
        if (shape_exp.numel() == 1) {
            float exp_val;
            cudaMemcpy(&exp_val, data_exp, sizeof(float), cudaMemcpyDeviceToHost);
            pow_kernel_scalar_exp<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                data_base, exp_val, data_out, total
            );
        } else if (shape_base == shape_exp) {
            pow_kernel_simple<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                data_base, data_exp, data_out, total
            );
        } else {
            pow_kernel_simple<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                data_base, data_exp, data_out, total
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

REGISTER_PLUGIN_SIMPLE(PowCUDAPlugin, "Pow", kPOW, CUDA)

}  // namespace operators
}  // namespace mini_infer
