#include "mini_infer/kernels/transpose.h"
#include <cstring>
#include <stdexcept>

namespace mini_infer {
namespace kernels {

/**
 * @brief CPU implementation of Transpose operations
 * 
 * TensorRT-style: Auto-register at program startup
 * 
 * Note: Currently not used in Conv2D (uses per-batch GEMM on CPU).
 * Kept for potential future GPU implementation where batched GEMM
 * requires CNHW to NCHW transpose.
 */

namespace cpu {

/**
 * @brief Transpose from CNHW to NCHW layout
 * 
 * Input:  [C, N*H*W] where data is organized as [C][N0_hw, N1_hw, ...]
 * Output: [N, C, H*W] where data is organized as [N][C0_hw, C1_hw, ...]
 * 
 * Algorithm:
 *   output[n, c, hw] = input[c, n*HW + hw]
 *   
 * Memory access pattern:
 *   - Input: strided access (stride = N*H*W)
 *   - Output: sequential writes
 */
template<typename T>
void transpose_CNHW_to_NCHW_impl(
    const T* input,
    T* output,
    int C,
    int N,
    int H,
    int W) {
    
    int spatial_size = H * W;
    int input_stride = N * spatial_size;  // Stride between channels in input
    
    // Iterate in output order for better write locality
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            // Source: input[c, n*spatial_size : (n+1)*spatial_size]
            // Dest:   output[n, c, :]
            const T* src = input + c * input_stride + n * spatial_size;
            T* dst = output + (n * C + c) * spatial_size;
            
            // Copy spatial dimensions using memcpy (optimized)
            std::memcpy(dst, src, spatial_size * sizeof(T));
        }
    }
}

// ============================================================================
// Explicit Registration Function 
// ============================================================================

void register_transpose_kernels() {
    // CPU availability checker (inline lambda)
    auto is_cpu_available = []() { return true; };
    
    // Register CPU Transpose implementation for float
    TransposeRegistry_CNHW_to_NCHW<float>::instance().register_kernel(
        KernelBackend::CPU,
        transpose_CNHW_to_NCHW_impl<float>,
        is_cpu_available,
        100  // Priority: CPU is baseline
    );
    
    // Register CPU Transpose implementation for int32_t
    TransposeRegistry_CNHW_to_NCHW<int32_t>::instance().register_kernel(
        KernelBackend::CPU,
        transpose_CNHW_to_NCHW_impl<int32_t>,
        is_cpu_available,
        100  // Priority: CPU is baseline
    );
}

} // namespace cpu

} // namespace kernels
} // namespace mini_infer
