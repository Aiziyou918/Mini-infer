#pragma once

#include "mini_infer/kernels/kernel_types.h"
#include "mini_infer/kernels/kernel_base.h"
#include "mini_infer/kernels/kernel_registry_template.h"
#include <stdexcept>

namespace mini_infer {
namespace kernels {

/**
 * @brief Im2Col function signature
 */
template<typename T>
using Im2ColFunc = void(*)(const T*, T*, int, int, int, int, int, 
                          int, int, int, int, int, int, int, int);

/**
 * @brief Im2Col Registry
 * 
 * Using template-based registry to eliminate code duplication.
 */
DEFINE_REGISTRY_ALIAS(Im2ColRegistry, Im2ColFunc);

/**
 * @brief Im2Col Kernel Interface
 * 
 * TensorRT-style: Auto-dispatch to best available implementation.
 */
class Im2ColKernel {
public:
    /**
     * @brief Im2col transformation for convolution
     * 
     * TensorRT-style: Automatically dispatches to best available implementation.
     * 
     * @param backend Backend selection:
     *   - AUTO (default): Auto-select best available
     *   - CPU/CPU_AVX2/etc: Force specific backend
     */
    template<typename T>
    static void im2col(
        const T* input,
        T* col_buffer,
        int channels,
        int height,
        int width,
        int kernel_h,
        int kernel_w,
        int stride_h,
        int stride_w,
        int padding_h,
        int padding_w,
        int dilation_h,
        int dilation_w,
        int out_height,
        int out_width,
        KernelBackend backend = KernelBackend::AUTO
    ) {
        Im2ColFunc<T> func = nullptr;
        
        // Get kernel from registry
        if (backend == KernelBackend::AUTO) {
            // Auto-select best available
            func = Im2ColRegistry<T>::instance().get_best_kernel();
        } else {
            // Use specific backend
            func = Im2ColRegistry<T>::instance().get_kernel(backend);
        }
        
        if (func) {
            func(input, col_buffer, channels, height, width,
                kernel_h, kernel_w, stride_h, stride_w,
                padding_h, padding_w, dilation_h, dilation_w,
                out_height, out_width);
        } else {
            throw std::runtime_error("No Im2Col kernel available for requested backend");
        }
    }
    
    /**
     * @brief Get the best available backend
     */
    DEFINE_BEST_BACKEND_GETTER(get_best_backend, Im2ColRegistry)
    
    /**
     * @brief Check if specific backend is available
     */
    DEFINE_BACKEND_CHECKER(is_backend_available, Im2ColRegistry)
};

} // namespace kernels
} // namespace mini_infer
