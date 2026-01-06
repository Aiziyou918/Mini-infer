#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/operators/activation_type.h"
#include "mini_infer/core/buffer.h"
#include "mini_infer/kernels/gemm.h"
#include "mini_infer/kernels/im2col.h"
#include "mini_infer/kernels/bias.h"

namespace mini_infer {
namespace operators {

/**
 * @brief Conv2D CPU Plugin
 *
 * Performs: output = Activation(Conv2D(input, weight) + bias)
 * Uses im2col + GEMM approach for efficient computation.
 *
 * Input shapes:
 *   - input: [N, C_in, H_in, W_in] (NCHW format)
 *   - weight: [C_out, C_in/groups, kernel_h, kernel_w]
 *   - bias (optional): [C_out]
 *
 * Output shape:
 *   - output: [N, C_out, H_out, W_out]
 */
class Conv2DCPUPlugin : public CPUPlugin<Conv2DCPUPlugin, Conv2DParam> {
public:
    Conv2DCPUPlugin() {
        param_ = std::make_shared<Conv2DParam>();
    }
    ~Conv2DCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Conv";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kCONVOLUTION;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return param_ && param_->use_bias ? 3 : 2;
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {

        if (input_shapes.size() < 2) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_shape = input_shapes[0];   // [N, C_in, H_in, W_in]
        const auto& weight_shape = input_shapes[1];  // [C_out, C_in/groups, kernel_h, kernel_w]

        if (input_shape.ndim() != 4 || weight_shape.ndim() != 4) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        int64_t N = input_shape[0];
        int64_t C_in = input_shape[1];
        int64_t H_in = input_shape[2];
        int64_t W_in = input_shape[3];

        int64_t C_out = weight_shape[0];
        int64_t C_in_per_group = weight_shape[1];
        int64_t kernel_h = weight_shape[2];
        int64_t kernel_w = weight_shape[3];

        // Validate groups
        int groups = param_ ? param_->groups : 1;
        if (groups != 1) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }
        if (C_in != C_in_per_group * groups) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Validate bias shape if needed
        if (param_ && param_->use_bias && input_shapes.size() > 2) {
            const auto& bias_shape = input_shapes[2];
            if (bias_shape.ndim() != 1 || bias_shape[0] != C_out) {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
        }

        // Get parameters
        int pad_h = param_ ? param_->padding_h : 0;
        int pad_w = param_ ? param_->padding_w : 0;
        int stride_h = param_ ? param_->stride_h : 1;
        int stride_w = param_ ? param_->stride_w : 1;
        int dilation_h = param_ ? param_->dilation_h : 1;
        int dilation_w = param_ ? param_->dilation_w : 1;

        // Calculate output dimensions
        int64_t H_out = (H_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        int64_t W_out = (W_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

        output_shapes.clear();
        output_shapes.push_back(core::Shape({N, C_out, H_out, W_out}));
        return core::Status::SUCCESS;
    }

    size_t get_workspace_size(
        const std::vector<core::Shape>& input_shapes) const noexcept override {

        if (input_shapes.size() < 2) {
            return 0;
        }

        const auto& input_shape = input_shapes[0];
        const auto& weight_shape = input_shapes[1];

        if (input_shape.ndim() != 4 || weight_shape.ndim() != 4) {
            return 0;
        }

        int64_t C_in = input_shape[1];
        int64_t H_in = input_shape[2];
        int64_t W_in = input_shape[3];

        int64_t kernel_h = weight_shape[2];
        int64_t kernel_w = weight_shape[3];

        int pad_h = param_ ? param_->padding_h : 0;
        int pad_w = param_ ? param_->padding_w : 0;
        int stride_h = param_ ? param_->stride_h : 1;
        int stride_w = param_ ? param_->stride_w : 1;
        int dilation_h = param_ ? param_->dilation_h : 1;
        int dilation_w = param_ ? param_->dilation_w : 1;

        int64_t H_out = (H_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        int64_t W_out = (W_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

        // im2col buffer size per batch
        int64_t col_size = C_in * kernel_h * kernel_w * H_out * W_out;
        return static_cast<size_t>(col_size * sizeof(float));
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;

        if (inputs.size() < 2 || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input = inputs[0];
        const auto& weight = inputs[1];
        auto& output = outputs[0];

        if (!input || !weight || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        if (input->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& input_shape = input->shape();
        const auto& weight_shape = weight->shape();
        const auto& output_shape = output->shape();

        int N = static_cast<int>(input_shape[0]);
        int C_in = static_cast<int>(input_shape[1]);
        int H_in = static_cast<int>(input_shape[2]);
        int W_in = static_cast<int>(input_shape[3]);

        int kernel_h = static_cast<int>(weight_shape[2]);
        int kernel_w = static_cast<int>(weight_shape[3]);

        int C_out = static_cast<int>(output_shape[1]);
        int H_out = static_cast<int>(output_shape[2]);
        int W_out = static_cast<int>(output_shape[3]);

        const float* input_data = static_cast<const float*>(input->data());
        const float* weight_data = static_cast<const float*>(weight->data());
        float* output_data = static_cast<float*>(output->data());

        // im2col + GEMM approach
        int spatial_size = H_out * W_out;
        int col_size_per_batch = C_in * kernel_h * kernel_w * spatial_size;
        core::Buffer<float> col_buffer(col_size_per_batch);

        int M = C_out;
        int N_gemm = spatial_size;
        int K = C_in * kernel_h * kernel_w;

        for (int n = 0; n < N; ++n) {
            const float* input_n = input_data + n * C_in * H_in * W_in;
            float* output_n = output_data + n * C_out * spatial_size;

            // im2col transformation
            kernels::Im2ColKernel::im2col<float>(
                input_n, col_buffer.data(),
                C_in, H_in, W_in,
                kernel_h, kernel_w,
                param_->stride_h, param_->stride_w,
                param_->padding_h, param_->padding_w,
                param_->dilation_h, param_->dilation_w,
                H_out, W_out,
                kernels::KernelBackend::CPU
            );

            // GEMM: output = weight * col
            kernels::GEMMKernel::gemm_nn<float>(
                weight_data, col_buffer.data(), output_n,
                M, N_gemm, K,
                kernels::KernelBackend::CPU
            );
        }

        // Add bias if present
        if (param_ && param_->use_bias && inputs.size() > 2 && inputs[2]) {
            const float* bias_data = static_cast<const float*>(inputs[2]->data());
            kernels::BiasKernel::add_channel_bias<float>(
                output_data, bias_data,
                N, C_out, H_out * W_out,
                kernels::KernelBackend::CPU
            );
        }

        // Apply activation if enabled
        if (param_ && param_->activation != ActivationType::NONE) {
            int total_elements = N * C_out * H_out * W_out;
            for (int i = 0; i < total_elements; ++i) {
                if (param_->activation == ActivationType::RELU) {
                    output_data[i] = output_data[i] > 0.0f ? output_data[i] : 0.0f;
                }
                // Add other activation types as needed
            }
        }

        return core::Status::SUCCESS;
    }
};

// Define creator and register plugin
REGISTER_PLUGIN_SIMPLE(Conv2DCPUPlugin, "Conv", kCONVOLUTION, CPU)

}  // namespace operators
}  // namespace mini_infer
