#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <algorithm>
#include <string>

namespace mini_infer {
namespace operators {

namespace {

// LayerNorm kernel - one block per sample
__global__ void layernorm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int64_t inner_size,
    float epsilon
) {
    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sum_sq = shared + blockDim.x;

    int64_t sample_idx = blockIdx.x;
    const float* in_ptr = input + sample_idx * inner_size;
    float* out_ptr = output + sample_idx * inner_size;

    // Compute partial sums
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int64_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        float val = in_ptr[i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Compute mean and variance
    float mean = s_sum[0] / static_cast<float>(inner_size);
    float var = s_sum_sq[0] / static_cast<float>(inner_size) - mean * mean;
    float inv_std = rsqrtf(var + epsilon);

    // Normalize
    for (int64_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        float normalized = (in_ptr[i] - mean) * inv_std;
        if (gamma) normalized *= gamma[i];
        if (beta) normalized += beta[i];
        out_ptr[i] = normalized;
    }
}

}  // namespace

/**
 * @brief LayerNorm CUDA Plugin
 */
class LayerNormCUDAPlugin : public CUDAPlugin<LayerNormCUDAPlugin, LayerNormParam> {
public:
    LayerNormCUDAPlugin() {
        param_ = std::make_shared<LayerNormParam>();
    }
    ~LayerNormCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "LayerNormalization";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kLAYER_NORM;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return 3;
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {
        if (input_shapes.empty()) {
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

        if (inputs.size() < 1 || outputs.size() != 1) {
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

        const auto& shape = input->shape();
        const auto& dims = shape.dims();
        const size_t ndim = dims.size();

        int64_t axis = param_ ? param_->axis : -1;
        if (axis < 0) axis += static_cast<int64_t>(ndim);
        if (axis < 0 || axis >= static_cast<int64_t>(ndim)) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        float epsilon = param_ ? param_->epsilon : 1e-5f;

        int64_t outer_size = 1;
        int64_t inner_size = 1;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
            outer_size *= dims[i];
        }
        for (size_t i = static_cast<size_t>(axis); i < ndim; ++i) {
            inner_size *= dims[i];
        }

        const float* gamma = nullptr;
        const float* beta = nullptr;
        if (inputs.size() > 1 && inputs[1]) {
            gamma = static_cast<const float*>(inputs[1]->data());
        }
        if (inputs.size() > 2 && inputs[2]) {
            beta = static_cast<const float*>(inputs[2]->data());
        }

        const int threads = 256;
        const int blocks = static_cast<int>(outer_size);
        const size_t shared_mem = 2 * threads * sizeof(float);

        layernorm_kernel<<<blocks, threads, shared_mem, cuda_ctx->stream()>>>(
            static_cast<const float*>(input->data()),
            gamma,
            beta,
            static_cast<float*>(output->data()),
            inner_size,
            epsilon
        );

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(LayerNormCUDAPlugin, "LayerNormalization", kLAYER_NORM, CUDA)

}  // namespace operators
}  // namespace mini_infer
