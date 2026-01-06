#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <string>
#include <cfloat>
#include <algorithm>

namespace mini_infer {
namespace operators {

namespace {

/**
 * @brief Softmax CUDA kernel (numerically stable)
 *
 * Each block handles one "row" (outer * inner combination).
 * Uses shared memory for reduction operations.
 */
__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer,
    int dim,
    int inner
) {
    extern __shared__ float shared[];

    int row_idx = blockIdx.x;
    if (row_idx >= outer * inner) return;

    int o = row_idx / inner;
    int i = row_idx % inner;

    int tid = threadIdx.x;
    int base = o * dim * inner + i;

    // Step 1: Find max value (parallel reduction)
    float local_max = -FLT_MAX;
    for (int d = tid; d < dim; d += blockDim.x) {
        float val = input[base + d * inner];
        local_max = fmaxf(local_max, val);
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    float max_val = shared[0];
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float val = expf(input[base + d * inner] - max_val);
        output[base + d * inner] = val;
        local_sum += val;
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float sum_val = shared[0];
    __syncthreads();

    // Step 3: Normalize
    float inv_sum = (sum_val > 0.0f) ? (1.0f / sum_val) : 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        output[base + d * inner] *= inv_sum;
    }
}

/**
 * @brief Simple softmax kernel for small dimensions
 */
__global__ void softmax_kernel_simple(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer,
    int dim,
    int inner
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;

    if (idx >= total) return;

    int o = idx / inner;
    int i = idx % inner;
    int base = o * dim * inner + i;

    float max_val = -FLT_MAX;
    for (int d = 0; d < dim; ++d) {
        max_val = fmaxf(max_val, input[base + d * inner]);
    }

    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float val = expf(input[base + d * inner] - max_val);
        output[base + d * inner] = val;
        sum += val;
    }

    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int d = 0; d < dim; ++d) {
        output[base + d * inner] *= inv_sum;
    }
}

}  // namespace

/**
 * @brief Softmax CUDA Plugin
 */
class SoftmaxCUDAPlugin : public CUDAPlugin<SoftmaxCUDAPlugin, SoftmaxParam> {
public:
    SoftmaxCUDAPlugin() {
        param_ = std::make_shared<SoftmaxParam>();
    }
    ~SoftmaxCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Softmax";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kSOFTMAX;
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

        const auto& shape = input->shape();
        if (shape.ndim() == 0) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto dims = shape.dims();
        int axis = param_ ? param_->axis : -1;
        if (axis < 0) {
            axis += static_cast<int>(dims.size());
        }
        if (axis < 0 || axis >= static_cast<int>(dims.size())) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        int outer = 1;
        for (int i = 0; i < axis; ++i) {
            outer *= static_cast<int>(dims[static_cast<size_t>(i)]);
        }
        int inner = 1;
        for (size_t i = static_cast<size_t>(axis + 1); i < dims.size(); ++i) {
            inner *= static_cast<int>(dims[i]);
        }
        int dim = static_cast<int>(dims[static_cast<size_t>(axis)]);

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        int total_rows = outer * inner;

        if (dim <= 32) {
            const int threads = 256;
            int blocks = (total_rows + threads - 1) / threads;
            softmax_kernel_simple<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                in_data, out_data, outer, dim, inner
            );
        } else {
            int threads = std::min(256, ((dim + 31) / 32) * 32);
            int shared_size = threads * sizeof(float);
            softmax_kernel<<<total_rows, threads, shared_size, cuda_ctx->stream()>>>(
                in_data, out_data, outer, dim, inner
            );
        }

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] Softmax plugin error: " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }
};

// Define creator and register plugin
REGISTER_PLUGIN_SIMPLE(SoftmaxCUDAPlugin, "Softmax", kSOFTMAX, CUDA)

}  // namespace operators
}  // namespace mini_infer
