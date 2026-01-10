#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace operators {

namespace {

__global__ void erf_kernel(
    const float* input,
    float* output,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = erff(input[idx]);
    }
}

__global__ void erf_kernel_vectorized(
    const float* input,
    float* output,
    int64_t n
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    int64_t vec_size = n / 4;
    const float4* in_vec = reinterpret_cast<const float4*>(input);
    float4* out_vec = reinterpret_cast<float4*>(output);

    for (int64_t i = tid; i < vec_size; i += stride) {
        float4 v = in_vec[i];
        float4 vo;
        vo.x = erff(v.x);
        vo.y = erff(v.y);
        vo.z = erff(v.z);
        vo.w = erff(v.w);
        out_vec[i] = vo;
    }

    int64_t remaining_start = vec_size * 4;
    for (int64_t i = remaining_start + tid; i < n; i += stride) {
        output[i] = erff(input[i]);
    }
}

}  // namespace

class ErfCUDAPlugin : public SimpleCUDAPlugin<ErfCUDAPlugin> {
public:
    ErfCUDAPlugin() = default;
    ~ErfCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Erf";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kERF;
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

        const int64_t total = input->shape().numel();
        const float* data_in = static_cast<const float*>(input->data());
        float* data_out = static_cast<float*>(output->data());

        const int threads = 256;

        if (total < 1024) {
            int blocks = (total + threads - 1) / threads;
            erf_kernel<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                data_in, data_out, total
            );
        } else {
            int blocks = std::min((total + threads * 4 - 1) / (threads * 4), (int64_t)2048);
            erf_kernel_vectorized<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                data_in, data_out, total
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

REGISTER_PLUGIN_SIMPLE(ErfCUDAPlugin, "Erf", kERF, CUDA)

}  // namespace operators
}  // namespace mini_infer
