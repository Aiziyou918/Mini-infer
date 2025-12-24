#include "mini_infer/kernels/kernel_registry.h"

#include <cstdint>

#include "mini_infer/kernels/bias.h"
#include "mini_infer/kernels/gemm.h"
#include "mini_infer/operators/linear.h"

namespace mini_infer {
namespace kernels {

namespace {

void linear_cpu_kernel(KernelContext* ctx) {
    if (!ctx || !ctx->inputs || !ctx->outputs) {
        return;
    }

    const auto& inputs = *ctx->inputs;
    auto& outputs = *ctx->outputs;
    const auto* param = ctx->param<operators::LinearParam>();
    if (!param) {
        return;
    }

    const size_t expected_inputs = param->use_bias ? 3 : 2;
    if (inputs.size() != expected_inputs || outputs.empty()) {
        return;
    }

    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    auto output = outputs[0];
    if (!input || !weight || !output) {
        return;
    }

    const auto& input_shape = input->shape();
    const auto& weight_shape = weight->shape();
    const auto& output_shape = output->shape();
    if (input_shape.ndim() < 2 || weight_shape.ndim() != 2 || output_shape.ndim() < 2) {
        return;
    }

    int batch_size = 1;
    for (size_t i = 0; i + 1 < input_shape.ndim(); ++i) {
        batch_size *= static_cast<int>(input_shape[i]);
    }

    const int in_features = static_cast<int>(input_shape[input_shape.ndim() - 1]);
    const int out_features = static_cast<int>(output_shape[output_shape.ndim() - 1]);

    const auto dtype = input->dtype();
    if (dtype == core::DataType::FLOAT32) {
        const float* input_data = static_cast<const float*>(input->data());
        const float* weight_data = static_cast<const float*>(weight->data());
        float* output_data = static_cast<float*>(output->data());

        GEMMKernel::gemm_nt<float>(input_data, weight_data, output_data,
                                   batch_size, out_features, in_features);

        if (param->use_bias) {
            const float* bias_data = static_cast<const float*>(inputs[2]->data());
            BiasKernel::add_channel_bias<float>(output_data, bias_data,
                                                batch_size, out_features, 1);
        }
    } else if (dtype == core::DataType::INT32) {
        const int32_t* input_data = static_cast<const int32_t*>(input->data());
        const int32_t* weight_data = static_cast<const int32_t*>(weight->data());
        int32_t* output_data = static_cast<int32_t*>(output->data());

        GEMMKernel::gemm_nt<int32_t>(input_data, weight_data, output_data,
                                     batch_size, out_features, in_features);

        if (param->use_bias) {
            const int32_t* bias_data = static_cast<const int32_t*>(inputs[2]->data());
            BiasKernel::add_channel_bias<int32_t>(output_data, bias_data,
                                                  batch_size, out_features, 1);
        }
    }
}

}  // namespace

struct LinearCPURegister {
    LinearCPURegister() {
        KernelRegistry::instance().register_kernel(core::OpType::kGEMM,
                                                   core::DeviceType::CPU,
                                                   core::DataType::FLOAT32, linear_cpu_kernel);
        KernelRegistry::instance().register_kernel(core::OpType::kGEMM,
                                                   core::DeviceType::CPU,
                                                   core::DataType::INT32, linear_cpu_kernel);
    }
};

static LinearCPURegister g_linear_cpu_register;

}  // namespace kernels
}  // namespace mini_infer
