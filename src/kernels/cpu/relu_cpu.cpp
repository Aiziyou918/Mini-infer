#include "mini_infer/kernels/kernel_registry.h"

#include <algorithm>

namespace mini_infer {
namespace kernels {

namespace {

void relu_cpu_f32(KernelContext* ctx) {
    if (!ctx || !ctx->inputs || !ctx->outputs) {
        return;
    }
    const auto& inputs = *ctx->inputs;
    auto& outputs = *ctx->outputs;
    if (inputs.empty() || outputs.empty() || !inputs[0] || !outputs[0]) {
        return;
    }

    const auto& shape = inputs[0]->shape();
    const size_t total = static_cast<size_t>(shape.numel());
    const float* in = static_cast<const float*>(inputs[0]->data());
    float* out = static_cast<float*>(outputs[0]->data());
    for (size_t i = 0; i < total; ++i) {
        out[i] = std::max(0.0f, in[i]);
    }
}

void relu_cpu_i32(KernelContext* ctx) {
    if (!ctx || !ctx->inputs || !ctx->outputs) {
        return;
    }
    const auto& inputs = *ctx->inputs;
    auto& outputs = *ctx->outputs;
    if (inputs.empty() || outputs.empty() || !inputs[0] || !outputs[0]) {
        return;
    }

    const auto& shape = inputs[0]->shape();
    const size_t total = static_cast<size_t>(shape.numel());
    const int32_t* in = static_cast<const int32_t*>(inputs[0]->data());
    int32_t* out = static_cast<int32_t*>(outputs[0]->data());
    for (size_t i = 0; i < total; ++i) {
        out[i] = std::max(0, in[i]);
    }
}

}  // namespace

struct ReluCpuRegister {
    ReluCpuRegister() {
        KernelRegistry::instance().register_kernel(core::OpType::kRELU,
                                                   core::DeviceType::CPU,
                                                   core::DataType::FLOAT32, relu_cpu_f32);
        KernelRegistry::instance().register_kernel(core::OpType::kRELU,
                                                   core::DeviceType::CPU,
                                                   core::DataType::INT32, relu_cpu_i32);
    }
};

static ReluCpuRegister g_relu_cpu_register;

}  // namespace kernels
}  // namespace mini_infer
