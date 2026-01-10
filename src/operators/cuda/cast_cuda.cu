#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace operators {

namespace {

template<typename SrcT, typename DstT>
__global__ void cast_kernel(
    const SrcT* input,
    DstT* output,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = static_cast<DstT>(input[idx]);
    }
}

}  // namespace

class CastCUDAPlugin : public SimpleCUDAPlugin<CastCUDAPlugin> {
public:
    CastCUDAPlugin() = default;
    ~CastCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Cast";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kCAST;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return 1;
    }

    void set_param(std::shared_ptr<PluginParam> param) override {
        param_ = std::dynamic_pointer_cast<CastParam>(param);
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

        output_shapes.clear();
        output_shapes.push_back(input_shapes[0]);
        return core::Status::SUCCESS;
    }

    core::Status infer_output_metadata(
        const std::vector<core::Shape>& input_shapes,
        const std::vector<core::DataType>& input_dtypes,
        std::vector<core::Shape>& output_shapes,
        std::vector<core::DataType>& output_dtypes) const override {
        (void)input_dtypes;

        auto status = infer_output_shapes(input_shapes, output_shapes);
        if (status != core::Status::SUCCESS) {
            return status;
        }

        output_dtypes.clear();
        core::DataType target_dtype = core::DataType::FLOAT32;
        if (param_) {
            switch (param_->to_dtype) {
                case 1: target_dtype = core::DataType::FLOAT32; break;
                case 6: target_dtype = core::DataType::INT32; break;
                case 7: target_dtype = core::DataType::INT64; break;
                default: target_dtype = core::DataType::FLOAT32; break;
            }
        }
        output_dtypes.push_back(target_dtype);
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

        const int64_t total = input->shape().numel();
        core::DataType src_dtype = input->dtype();
        core::DataType dst_dtype = output->dtype();

        const int threads = 256;
        int blocks = (total + threads - 1) / threads;

        // Fast path: same dtype
        if (src_dtype == dst_dtype) {
            cudaMemcpyAsync(output->data(), input->data(),
                           total * input->element_size(),
                           cudaMemcpyDeviceToDevice, cuda_ctx->stream());
        } else if (src_dtype == core::DataType::FLOAT32 && dst_dtype == core::DataType::INT64) {
            cast_kernel<float, int64_t><<<blocks, threads, 0, cuda_ctx->stream()>>>(
                static_cast<const float*>(input->data()),
                static_cast<int64_t*>(output->data()),
                total
            );
        } else if (src_dtype == core::DataType::INT64 && dst_dtype == core::DataType::FLOAT32) {
            cast_kernel<int64_t, float><<<blocks, threads, 0, cuda_ctx->stream()>>>(
                static_cast<const int64_t*>(input->data()),
                static_cast<float*>(output->data()),
                total
            );
        } else if (src_dtype == core::DataType::INT32 && dst_dtype == core::DataType::FLOAT32) {
            cast_kernel<int32_t, float><<<blocks, threads, 0, cuda_ctx->stream()>>>(
                static_cast<const int32_t*>(input->data()),
                static_cast<float*>(output->data()),
                total
            );
        } else if (src_dtype == core::DataType::FLOAT32 && dst_dtype == core::DataType::INT32) {
            cast_kernel<float, int32_t><<<blocks, threads, 0, cuda_ctx->stream()>>>(
                static_cast<const float*>(input->data()),
                static_cast<int32_t*>(output->data()),
                total
            );
        } else {
            // Fallback: copy if same element size
            if (input->element_size() == output->element_size()) {
                cudaMemcpyAsync(output->data(), input->data(),
                               total * input->element_size(),
                               cudaMemcpyDeviceToDevice, cuda_ctx->stream());
            } else {
                return core::Status::ERROR_NOT_IMPLEMENTED;
            }
        }

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }

private:
    std::shared_ptr<CastParam> param_;
};

REGISTER_PLUGIN_SIMPLE(CastCUDAPlugin, "Cast", kCAST, CUDA)

}  // namespace operators
}  // namespace mini_infer
