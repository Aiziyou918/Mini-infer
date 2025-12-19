#include "mini_infer/kernels/kernel_registry.h"

#include <cstring>

namespace mini_infer {
namespace kernels {

namespace {

void reshape_cpu(KernelContext* ctx) {
    if (!ctx || !ctx->inputs || !ctx->outputs) {
        return;
    }
    const auto& inputs = *ctx->inputs;
    auto& outputs = *ctx->outputs;
    if (inputs.empty() || outputs.empty() || !inputs[0] || !outputs[0]) {
        return;
    }

    const auto* src = inputs[0]->data();
    auto* dst = outputs[0]->data();
    if (src && dst && src != dst) {
        std::memcpy(dst, src, inputs[0]->size_in_bytes());
    }
}

void register_reshape_dtype(core::DataType dtype) {
    KernelRegistry::instance().register_kernel(core::OpType::kRESHAPE,
                                               core::DeviceType::CPU, dtype, reshape_cpu);
}

}  // namespace

struct ReshapeCpuRegister {
    ReshapeCpuRegister() {
        register_reshape_dtype(core::DataType::FLOAT32);
        register_reshape_dtype(core::DataType::FLOAT16);
        register_reshape_dtype(core::DataType::INT32);
        register_reshape_dtype(core::DataType::INT64);
        register_reshape_dtype(core::DataType::INT8);
        register_reshape_dtype(core::DataType::UINT8);
        register_reshape_dtype(core::DataType::BOOL);
    }
};

static ReshapeCpuRegister g_reshape_cpu_register;

}  // namespace kernels
}  // namespace mini_infer
