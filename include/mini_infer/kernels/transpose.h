#pragma once

#include "mini_infer/kernels/kernel_types.h"
#include "mini_infer/kernels/kernel_base.h"
#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/kernels/kernel_registry_template.h"
#include <stdexcept>

namespace mini_infer {
namespace kernels {

/**
 * @brief Transpose function signature for CNHW to NCHW
 */
template<typename T>
using TransposeFunc_CNHW_to_NCHW = void(*)(const T*, T*, int, int, int, int);

/**
 * @brief Transpose Registry for CNHW to NCHW
 */
DEFINE_REGISTRY_ALIAS(TransposeRegistry_CNHW_to_NCHW, TransposeFunc_CNHW_to_NCHW);

/**
 * @brief Transpose Kernel Interface
 * 
 * TensorRT-style: Provides efficient layout transformations
 * Required for batched GEMM convolution outputs
 */
class TransposeKernel {
public:
    /**
     * @brief Transpose from CNHW to NCHW layout
     * 
     * Used in Conv2D to transform batched GEMM output to correct layout.
     * 
     * Input layout (CNHW):  [C, N*H*W] -> [C][batch0_hw][batch1_hw]...
     * Output layout (NCHW): [N, C, H*W] -> [N][C0_hw][C1_hw]...
     * 
     * @tparam T Data type
     * @param input Input data in CNHW layout [C, N, H, W]
     * @param output Output data in NCHW layout [N, C, H, W]
     * @param C Number of channels
     * @param N Number of batches
     * @param H Height
     * @param W Width
     * @param backend Backend selection:
     *   - AUTO (default): Auto-select best available
     *   - CPU/CPU_AVX2/etc: Force specific backend
     * 
     * Example:
     *   Input:  C0[n0_hw, n1_hw], C1[n0_hw, n1_hw]
     *   Output: N0[c0_hw, c1_hw], N1[c0_hw, c1_hw]
     */
    template<typename T>
    static void transpose_CNHW_to_NCHW(
        const T* input,
        T* output,
        int C,
        int N,
        int H,
        int W,
        KernelBackend backend = KernelBackend::AUTO
    ) {
        // Ensure kernels are initialized
        KernelRegistryInitializer::initialize();
        
        TransposeFunc_CNHW_to_NCHW<T> func = nullptr;
        
        // Get kernel from registry
        if (backend == KernelBackend::AUTO) {
            func = TransposeRegistry_CNHW_to_NCHW<T>::instance().get_best_kernel();
        } else {
            func = TransposeRegistry_CNHW_to_NCHW<T>::instance().get_kernel(backend);
        }
        
        if (func) {
            func(input, output, C, N, H, W);
        } else {
            throw std::runtime_error("No Transpose kernel available for requested backend");
        }
    }
    
    /**
     * @brief Get the best available backend
     */
    DEFINE_BEST_BACKEND_GETTER(get_best_backend, TransposeRegistry_CNHW_to_NCHW)
    
    /**
     * @brief Check if specific backend is available
     */
    DEFINE_BACKEND_CHECKER(is_backend_available, TransposeRegistry_CNHW_to_NCHW)
};

} // namespace kernels
} // namespace mini_infer
