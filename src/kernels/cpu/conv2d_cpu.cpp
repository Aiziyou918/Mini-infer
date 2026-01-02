#include "mini_infer/kernels/kernel_registry.h"

#include "mini_infer/core/buffer.h"
#include "mini_infer/kernels/bias.h"
#include "mini_infer/kernels/gemm.h"
#include "mini_infer/kernels/im2col.h"
#include "mini_infer/operators/activation_type.h"
#include "mini_infer/operators/conv2d.h"

namespace mini_infer {
namespace kernels {

namespace {

void conv2d_cpu_kernel(KernelContext* ctx) {
    if (!ctx || !ctx->inputs || !ctx->outputs) {
        return;
    }

    const auto& inputs = *ctx->inputs;
    auto& outputs = *ctx->outputs;
    const auto* param = ctx->param<operators::Conv2DParam>();
    if (!param) {
        return;
    }

    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    auto output = outputs[0];

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

    const auto dtype = input->dtype();
    if (dtype != core::DataType::FLOAT32) {
        return;
    }

    const float* input_data = static_cast<const float*>(input->data());
    const float* weight_data = static_cast<const float*>(weight->data());
    float* output_data = static_cast<float*>(output->data());

    int spatial_size = H_out * W_out;
    int col_size_per_batch = C_in * kernel_h * kernel_w * spatial_size;
    core::Buffer<float> col_buffer(col_size_per_batch);

    int M = C_out;
    int N_gemm = spatial_size;
    int K = C_in * kernel_h * kernel_w;

    for (int n = 0; n < N; ++n) {
        const float* input_n = input_data + n * C_in * H_in * W_in;
        float* output_n = output_data + n * C_out * spatial_size;

        kernels::Im2ColKernel::im2col<float>(
            input_n, col_buffer.data(), C_in, H_in, W_in, kernel_h, kernel_w, param->stride_h,
            param->stride_w, param->padding_h, param->padding_w, param->dilation_h,
            param->dilation_w, H_out, W_out, KernelBackend::CPU);

        kernels::GEMMKernel::gemm_nn<float>(weight_data, col_buffer.data(), output_n, M, N_gemm, K,
                                             KernelBackend::CPU);
    }

    if (param->use_bias) {
        const float* bias_data = static_cast<const float*>(inputs[2]->data());
        kernels::BiasKernel::add_channel_bias<float>(output_data, bias_data,
                                                     N,             // batch_size
                                                     C_out,         // channels
                                                     H_out * W_out,  // spatial_size
                                                     KernelBackend::CPU
        );
    }

    if (param->activation.is_enabled()) {
        int total_elements = N * C_out * H_out * W_out;
        for (int i = 0; i < total_elements; ++i) {
            output_data[i] = operators::apply_activation(output_data[i], param->activation);
        }
    }
}

}  // namespace

struct Conv2DCPURegister {
    Conv2DCPURegister() {
        KernelRegistry::instance().register_kernel(core::OpType::kCONVOLUTION,
                                                   core::DeviceType::CPU,
                                                   core::DataType::FLOAT32, conv2d_cpu_kernel);
    }
};

static Conv2DCPURegister g_conv2d_cpu_register;

}  // namespace kernels
}  // namespace mini_infer
