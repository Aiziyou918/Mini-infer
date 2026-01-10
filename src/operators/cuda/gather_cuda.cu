#include "mini_infer/operators/cuda_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <algorithm>
#include <string>

namespace mini_infer {
namespace operators {

namespace {

__global__ void gather_kernel_int64(
    const float* data,
    const int64_t* indices,
    float* output,
    int64_t outer_size,
    int64_t axis_size,
    int64_t inner_size,
    int64_t num_indices
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * num_indices * inner_size;
    if (idx >= total) return;

    int64_t inner_idx = idx % inner_size;
    int64_t indices_idx = (idx / inner_size) % num_indices;
    int64_t outer_idx = idx / (inner_size * num_indices);

    int64_t gather_idx = indices[indices_idx];
    if (gather_idx < 0) gather_idx += axis_size;

    int64_t src_idx = (outer_idx * axis_size + gather_idx) * inner_size + inner_idx;
    output[idx] = data[src_idx];
}

__global__ void gather_kernel_int32(
    const float* data,
    const int32_t* indices,
    float* output,
    int64_t outer_size,
    int64_t axis_size,
    int64_t inner_size,
    int64_t num_indices
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * num_indices * inner_size;
    if (idx >= total) return;

    int64_t inner_idx = idx % inner_size;
    int64_t indices_idx = (idx / inner_size) % num_indices;
    int64_t outer_idx = idx / (inner_size * num_indices);

    int64_t gather_idx = indices[indices_idx];
    if (gather_idx < 0) gather_idx += axis_size;

    int64_t src_idx = (outer_idx * axis_size + gather_idx) * inner_size + inner_idx;
    output[idx] = data[src_idx];
}

}  // namespace

/**
 * @brief Gather CUDA Plugin
 */
class GatherCUDAPlugin : public CUDAPlugin<GatherCUDAPlugin, GatherParam> {
public:
    GatherCUDAPlugin() {
        param_ = std::make_shared<GatherParam>();
    }
    ~GatherCUDAPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Gather";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kGATHER;
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

        const auto& data_dims = input_shapes[0].dims();
        const auto& indices_dims = input_shapes[1].dims();

        int64_t axis = param_ ? param_->axis : 0;
        int64_t data_ndim = static_cast<int64_t>(data_dims.size());
        if (axis < 0) axis += data_ndim;
        if (axis < 0 || axis >= data_ndim) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        std::vector<int64_t> out_dims;
        for (int64_t i = 0; i < axis; ++i) {
            out_dims.push_back(data_dims[i]);
        }
        for (const auto& d : indices_dims) {
            out_dims.push_back(d);
        }
        for (int64_t i = axis + 1; i < data_ndim; ++i) {
            out_dims.push_back(data_dims[i]);
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

        const auto& data = inputs[0];
        const auto& indices = inputs[1];
        auto& output = outputs[0];

        if (!data || !indices || !output || !context.device_context) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(
            context.device_context
        );
        if (!cuda_ctx) {
            return core::Status::ERROR_BACKEND;
        }

        if (data->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& data_dims = data->shape().dims();
        const int64_t data_ndim = static_cast<int64_t>(data_dims.size());

        int64_t axis = param_ ? param_->axis : 0;
        if (axis < 0) axis += data_ndim;

        int64_t outer_size = 1;
        int64_t inner_size = 1;
        for (int64_t i = 0; i < axis; ++i) {
            outer_size *= data_dims[i];
        }
        for (int64_t i = axis + 1; i < data_ndim; ++i) {
            inner_size *= data_dims[i];
        }

        const int64_t axis_size = data_dims[axis];
        const int64_t num_indices = indices->shape().numel();
        const int64_t total = outer_size * num_indices * inner_size;

        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;

        if (indices->dtype() == core::DataType::INT64) {
            gather_kernel_int64<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                static_cast<const float*>(data->data()),
                static_cast<const int64_t*>(indices->data()),
                static_cast<float*>(output->data()),
                outer_size, axis_size, inner_size, num_indices
            );
        } else if (indices->dtype() == core::DataType::INT32) {
            gather_kernel_int32<<<blocks, threads, 0, cuda_ctx->stream()>>>(
                static_cast<const float*>(data->data()),
                static_cast<const int32_t*>(indices->data()),
                static_cast<float*>(output->data()),
                outer_size, axis_size, inner_size, num_indices
            );
        } else {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDA] " + std::string(cudaGetErrorString(status)));
            return core::Status::ERROR_BACKEND;
        }

        return core::Status::SUCCESS;
    }
};

REGISTER_PLUGIN_SIMPLE(GatherCUDAPlugin, "Gather", kGATHER, CUDA)

}  // namespace operators
}  // namespace mini_infer
