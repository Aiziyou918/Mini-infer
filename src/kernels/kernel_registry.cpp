#include "mini_infer/kernels/kernel_registry.h"

namespace mini_infer {
namespace kernels {

// Forward declarations of registration functions
namespace cpu {
    void register_gemm_kernels();
    void register_im2col_kernels();
    void register_bias_kernels();
    void register_transpose_kernels();
}
void register_pooling_kernels();

bool KernelRegistryInitializer::initialized_ = false;

void KernelRegistryInitializer::initialize() {
    if (initialized_) {
        return;  // Already initialized
    }
    
    // Force registration of all CPU kernels
    cpu::register_gemm_kernels();
    cpu::register_im2col_kernels();
    cpu::register_bias_kernels();
    cpu::register_transpose_kernels();
    register_pooling_kernels();
    
    // Future: Register CUDA kernels
    // cuda::register_gemm_kernels();
    // cuda::register_im2col_kernels();
    
    initialized_ = true;
}

} // namespace kernels
} // namespace mini_infer
