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
 * @brief ReLU forward CUDA kernel (vectorized version)
 *
 * Computes: output[i] = max(0, input[i])
 *
 * Performance optimizations:
 * - Vectorized memory access using float4 (128-bit loads/stores)
 * - Grid-stride loop for handling arbitrary sizes
 * - Coalesced memory access pattern
 */
__global__ void relu_forward_kernel_vectorized(
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
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        val.z = fmaxf(0.0f, val.z);
        val.w = fmaxf(0.0f, val.w);
        output_vec[i] = val;
    }

    int64_t remaining_start = vec_size * 4;
    for (int64_t i = remaining_start + tid; i < n; i += stride) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

/**
 * @brief ReLU forward CUDA kernel (simple version for small tensors)
 */
__global__ void relu_forward_kernel_simple(
    const float* input,
    float* output,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

}  // namespace

/**
 * @brief ReLU CUDA Plugin
 *
 * Implements ReLU activation using hand-written CUDA kernels.
 * Automatically selects between vectorized and simple versions based on tensor size.
 */
class ReLUCUDAPlugin : public SimpleCUDAPlugin<ReLUCUDAPlugin> {
public:
    ReLUCUDAPlugin() = default;
    ~ReLUCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Relu";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kRELU;
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

        int64_t n = input->shape().numel();
        const int threads = 256;

        if (n < 1024) {
            int blocks = (n + threads - 1) / threads;
            relu_forward_kernel_simple<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                static_cast<const float*>(input->data()),
                static_cast<float*>(output->data()),
                n
            );
        } else {
            int blocks = std::min((n + threads * 4 - 1) / (threads * 4), (int64_t)2048);
            relu_forward_kernel_vectorized<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                static_cast<const float*>(input->data()),
                static_cast<float*>(output->data()),
                n
            );
        }

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] " + std::string(cudaGetErrorString(status)) +
                         " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }
};

// Define creator and register plugin
REGISTER_PLUGIN_SIMPLE(ReLUCUDAPlugin, "Relu", kRELU, CUDA)

}  // namespace operators
}  // namespace mini_infer
