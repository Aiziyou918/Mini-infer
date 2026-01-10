#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <algorithm>
#include <string>

namespace mini_infer {
namespace operators {

namespace {

__global__ void sqrt_kernel_vectorized(
    const float* input,
    float* output,
    int64_t n
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    int64_t vec_size = n / 4;
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    float4* output_vec = reinterpret_cast<float4*>(output);

    for (int64_t i = tid; i < vec_size; i += stride) {
        float4 val = input_vec[i];
        float4 out;
        out.x = sqrtf(val.x);
        out.y = sqrtf(val.y);
        out.z = sqrtf(val.z);
        out.w = sqrtf(val.w);
        output_vec[i] = out;
    }

    int64_t remaining_start = vec_size * 4;
    for (int64_t i = remaining_start + tid; i < n; i += stride) {
        output[i] = sqrtf(input[i]);
    }
}

__global__ void sqrt_kernel_simple(
    const float* input,
    float* output,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sqrtf(input[idx]);
    }
}

}  // namespace

/**
 * @brief Sqrt CUDA Plugin
 */
class SqrtCUDAPlugin : public SimpleCUDAPlugin<SqrtCUDAPlugin> {
public:
    SqrtCUDAPlugin() = default;
    ~SqrtCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Sqrt";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kSQRT;
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

        output_shapes.clear();
        output_shapes.push_back(input_shapes[0]);
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

        if (!input || !output || !context.device_context) {
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

        const int64_t n = input->shape().numel();
        const int threads = 256;

        if (n < 1024) {
            int blocks = (n + threads - 1) / threads;
            sqrt_kernel_simple<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                static_cast<const float*>(input->data()),
                static_cast<float*>(output->data()),
                n
            );
        } else {
            int blocks = std::min((n + threads * 4 - 1) / (threads * 4), (int64_t)2048);
            sqrt_kernel_vectorized<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                static_cast<const float*>(input->data()),
                static_cast<float*>(output->data()),
                n
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

REGISTER_PLUGIN_SIMPLE(SqrtCUDAPlugin, "Sqrt", kSQRT, CUDA)

}  // namespace operators
}  // namespace mini_infer
