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
 * @brief MaxPool2D CUDA kernel
 */
__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    int total = N * C * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int w_out = i % W_out;
        int h_out = (i / W_out) % H_out;
        int c = (i / (W_out * H_out)) % C;
        int n = i / (W_out * H_out * C);

        int h_start = h_out * stride_h - padding_h;
        int w_start = w_out * stride_w - padding_w;
        int h_end = min(h_start + kernel_h, H_in);
        int w_end = min(w_start + kernel_w, W_in);
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);

        float max_val = -FLT_MAX;
        const float* input_nc = input + (n * C + c) * H_in * W_in;

        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                max_val = fmaxf(max_val, input_nc[h * W_in + w]);
            }
        }

        output[i] = max_val;
    }
}

/**
 * @brief AvgPool2D CUDA kernel
 */
__global__ void avgpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    int total = N * C * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int w_out = i % W_out;
        int h_out = (i / W_out) % H_out;
        int c = (i / (W_out * H_out)) % C;
        int n = i / (W_out * H_out * C);

        int h_start = h_out * stride_h - padding_h;
        int w_start = w_out * stride_w - padding_w;
        int h_end = min(h_start + kernel_h, H_in);
        int w_end = min(w_start + kernel_w, W_in);
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);

        float sum = 0.0f;
        int count = 0;
        const float* input_nc = input + (n * C + c) * H_in * W_in;

        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                sum += input_nc[h * W_in + w];
                ++count;
            }
        }

        output[i] = (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
    }
}

}  // namespace

/**
 * @brief MaxPool CUDA Plugin
 */
class MaxPoolCUDAPlugin : public CUDAPlugin<MaxPoolCUDAPlugin, PoolingParam> {
public:
    MaxPoolCUDAPlugin() {
        param_ = std::make_shared<PoolingParam>();
        param_->type = PoolingType::MAX;
    }
    ~MaxPoolCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "MaxPool";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kMAX_POOL;
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
        if (input_shapes.size() != 1 || input_shapes[0].ndim() != 4) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_shape = input_shapes[0];
        int64_t N = input_shape[0];
        int64_t C = input_shape[1];
        int64_t H_in = input_shape[2];
        int64_t W_in = input_shape[3];

        int kh = param_ ? param_->kernel_h : 2;
        int kw = param_ ? param_->kernel_w : 2;
        int sh = param_ ? param_->stride_h : 2;
        int sw = param_ ? param_->stride_w : 2;
        int ph = param_ ? param_->padding_h : 0;
        int pw = param_ ? param_->padding_w : 0;

        int64_t H_out = (H_in + 2 * ph - kh) / sh + 1;
        int64_t W_out = (W_in + 2 * pw - kw) / sw + 1;

        output_shapes.clear();
        output_shapes.push_back(core::Shape({N, C, H_out, W_out}));
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

        const auto& input_shape = input->shape();
        const auto& output_shape = output->shape();

        int N = static_cast<int>(input_shape[0]);
        int C = static_cast<int>(input_shape[1]);
        int H_in = static_cast<int>(input_shape[2]);
        int W_in = static_cast<int>(input_shape[3]);
        int H_out = static_cast<int>(output_shape[2]);
        int W_out = static_cast<int>(output_shape[3]);

        int total = N * C * H_out * W_out;
        const int threads = 256;
        int blocks = std::min((total + threads - 1) / threads, 65535);

        maxpool2d_kernel<<<blocks, threads, 0, cuda_ctx->stream()>>>(
            static_cast<const float*>(input->data()),
            static_cast<float*>(output->data()),
            N, C, H_in, W_in, H_out, W_out,
            param_->kernel_h, param_->kernel_w,
            param_->stride_h, param_->stride_w,
            param_->padding_h, param_->padding_w
        );

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] MaxPool plugin error: " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }
};

/**
 * @brief AvgPool CUDA Plugin
 */
class AvgPoolCUDAPlugin : public CUDAPlugin<AvgPoolCUDAPlugin, PoolingParam> {
public:
    AvgPoolCUDAPlugin() {
        param_ = std::make_shared<PoolingParam>();
        param_->type = PoolingType::AVERAGE;
    }
    ~AvgPoolCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "AveragePool";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kAVERAGE_POOL;
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
        if (input_shapes.size() != 1 || input_shapes[0].ndim() != 4) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_shape = input_shapes[0];
        int64_t N = input_shape[0];
        int64_t C = input_shape[1];
        int64_t H_in = input_shape[2];
        int64_t W_in = input_shape[3];

        int kh = param_ ? param_->kernel_h : 2;
        int kw = param_ ? param_->kernel_w : 2;
        int sh = param_ ? param_->stride_h : 2;
        int sw = param_ ? param_->stride_w : 2;
        int ph = param_ ? param_->padding_h : 0;
        int pw = param_ ? param_->padding_w : 0;

        int64_t H_out = (H_in + 2 * ph - kh) / sh + 1;
        int64_t W_out = (W_in + 2 * pw - kw) / sw + 1;

        output_shapes.clear();
        output_shapes.push_back(core::Shape({N, C, H_out, W_out}));
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

        const auto& input_shape = input->shape();
        const auto& output_shape = output->shape();

        int N = static_cast<int>(input_shape[0]);
        int C = static_cast<int>(input_shape[1]);
        int H_in = static_cast<int>(input_shape[2]);
        int W_in = static_cast<int>(input_shape[3]);
        int H_out = static_cast<int>(output_shape[2]);
        int W_out = static_cast<int>(output_shape[3]);

        int total = N * C * H_out * W_out;
        const int threads = 256;
        int blocks = std::min((total + threads - 1) / threads, 65535);

        avgpool2d_kernel<<<blocks, threads, 0, cuda_ctx->stream()>>>(
            static_cast<const float*>(input->data()),
            static_cast<float*>(output->data()),
            N, C, H_in, W_in, H_out, W_out,
            param_->kernel_h, param_->kernel_w,
            param_->stride_h, param_->stride_w,
            param_->padding_h, param_->padding_w
        );

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] AvgPool plugin error: " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }
};

// Define creators and register plugins
REGISTER_PLUGIN_SIMPLE(MaxPoolCUDAPlugin, "MaxPool", kMAX_POOL, CUDA)
REGISTER_PLUGIN_SIMPLE(AvgPoolCUDAPlugin, "AveragePool", kAVERAGE_POOL, CUDA)

}  // namespace operators
}  // namespace mini_infer
