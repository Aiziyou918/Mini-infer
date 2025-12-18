#include "mini_infer/kernels/bias.h"
#include <stdexcept>

namespace mini_infer {
namespace kernels {

/**
 * @brief CPU implementation of Bias addition
 * 
 * TensorRT-style: Auto-register at program startup
 */

namespace cpu {

template<typename T>
void bias_impl(
    T* output,
    const T* bias,
    int batch_size,
    int channels,
    int spatial_size) {
    
    // Process each batch
    for (int b = 0; b < batch_size; ++b) {
        T* batch_output = output + b * channels * spatial_size;
        
        // Process each channel
        for (int c = 0; c < channels; ++c) {
            T bias_val = bias[c];
            T* channel_output = batch_output + c * spatial_size;
            
            // Add bias to all spatial positions
            for (int s = 0; s < spatial_size; ++s) {
                channel_output[s] += bias_val;
            }
        }
    }
}

// ============================================================================
// Explicit Registration Function 
// ============================================================================

void register_bias_kernels() {
    // CPU availability checker (inline lambda)
    auto is_cpu_available = []() { return true; };
    
    // Register CPU Bias implementation for float
    BiasRegistry<float>::instance().register_kernel(
        KernelBackend::CPU,
        bias_impl<float>,
        is_cpu_available,
        100  // Priority: CPU is baseline
    );
    
    // Register CPU Bias implementation for int32_t
    BiasRegistry<int32_t>::instance().register_kernel(
        KernelBackend::CPU,
        bias_impl<int32_t>,
        is_cpu_available,
        100  // Priority: CPU is baseline
    );
}

namespace {
struct BiasKernelsAutoRegister {
    BiasKernelsAutoRegister() { register_bias_kernels(); }
};
static BiasKernelsAutoRegister g_bias_kernels_auto_register;
}  // namespace

} // namespace cpu

} // namespace kernels
} // namespace mini_infer
