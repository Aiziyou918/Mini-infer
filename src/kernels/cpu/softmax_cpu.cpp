#include "mini_infer/kernels/kernel_registry.h"

#include <algorithm>
#include <cmath>

#include "mini_infer/operators/softmax.h"

namespace mini_infer {
namespace kernels {

namespace {

void softmax_cpu_f32(KernelContext* ctx) {
    if (!ctx || !ctx->inputs || !ctx->outputs) {
        return;
    }
    const auto& inputs = *ctx->inputs;
    auto& outputs = *ctx->outputs;
    if (inputs.empty() || outputs.empty() || !inputs[0] || !outputs[0]) {
        return;
    }

    const auto* param = ctx->param<operators::SoftmaxParam>();
    const auto& shape = inputs[0]->shape();
    if (!param || shape.ndim() == 0) {
        return;
    }

    const auto dims = shape.dims();
    int axis = param->axis;
    if (axis < 0) {
        axis += static_cast<int>(dims.size());
    }
    if (axis < 0 || axis >= static_cast<int>(dims.size())) {
        return;
    }

    size_t outer = 1;
    for (int i = 0; i < axis; ++i) {
        outer *= static_cast<size_t>(dims[static_cast<size_t>(i)]);
    }
    size_t inner = 1;
    for (size_t i = static_cast<size_t>(axis + 1); i < dims.size(); ++i) {
        inner *= static_cast<size_t>(dims[i]);
    }
    const size_t dim = static_cast<size_t>(dims[static_cast<size_t>(axis)]);

    const float* input = static_cast<const float*>(inputs[0]->data());
    float* output = static_cast<float*>(outputs[0]->data());
    if (!input || !output) {
        return;
    }

    const size_t stride = inner;
    for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < inner; ++i) {
            const size_t base = o * dim * stride + i;

            float max_val = input[base];
            for (size_t d = 1; d < dim; ++d) {
                max_val = std::max(max_val, input[base + d * stride]);
            }

            float sum = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                float v = std::exp(input[base + d * stride] - max_val);
                output[base + d * stride] = v;
                sum += v;
            }

            if (sum == 0.0f) {
                continue;
            }
            const float inv_sum = 1.0f / sum;
            for (size_t d = 0; d < dim; ++d) {
                output[base + d * stride] *= inv_sum;
            }
        }
    }
}

}  // namespace

struct SoftmaxCpuRegister {
    SoftmaxCpuRegister() {
        KernelRegistry::instance().register_kernel(core::OpType::kSOFTMAX,
                                                   core::DeviceType::CPU,
                                                   core::DataType::FLOAT32, softmax_cpu_f32);
    }
};

static SoftmaxCpuRegister g_softmax_cpu_register;

}  // namespace kernels
}  // namespace mini_infer
