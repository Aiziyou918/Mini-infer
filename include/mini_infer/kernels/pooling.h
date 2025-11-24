#pragma once

#include "mini_infer/kernels/kernel_types.h"
#include "mini_infer/kernels/kernel_base.h"
#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/kernels/kernel_registry_template.h"
#include <stdexcept>

namespace mini_infer {
namespace kernels {

/**
 * @brief MaxPool2D kernel function signature
 * 
 * TensorRT-style: Function pointer for different backend implementations
 * 
 * @param input Input data [N, C, H_in, W_in]
 * @param output Output data [N, C, H_out, W_out]
 * @param N Batch size
 * @param C Number of channels
 * @param H_in Input height
 * @param W_in Input width
 * @param H_out Output height
 * @param W_out Output width
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param stride_h Stride in height
 * @param stride_w Stride in width
 * @param padding_h Padding in height
 * @param padding_w Padding in width
 */
template<typename T>
using MaxPool2DFunc = void (*)(
    const T* input,
    T* output,
    int N,
    int C,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w
);

/**
 * @brief AvgPool2D kernel function signature
 * 
 * TensorRT-style: Average pooling excludes padding (count_include_pad=false)
 * 
 * @param input Input data [N, C, H_in, W_in]
 * @param output Output data [N, C, H_out, W_out]
 * (same parameters as MaxPool2D)
 */
template<typename T>
using AvgPool2DFunc = void (*)(
    const T* input,
    T* output,
    int N,
    int C,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w
);

// ============================================================================
// Registry Definitions
// ============================================================================

/**
 * @brief Registry for MaxPool2D kernels
 * 
 * Using template-based registry to eliminate code duplication.
 */
DEFINE_REGISTRY_ALIAS(MaxPool2DRegistry, MaxPool2DFunc);

/**
 * @brief Registry for AvgPool2D kernels
 * 
 * Using template-based registry to eliminate code duplication.
 */
DEFINE_REGISTRY_ALIAS(AvgPool2DRegistry, AvgPool2DFunc);

/**
 * @brief Pooling Kernel Interface
 * 
 * TensorRT-style: Static dispatch to best available implementation
 */
class PoolingKernel {
public:
    /**
     * @brief MaxPool2D operation
     * 
     * TensorRT-style: Automatically dispatches to best available implementation.
     * 
     * @param backend Backend selection:
     *   - AUTO (default): Auto-select best available
     *   - CPU/CPU_AVX2/etc: Force specific backend
     */
    template<typename T>
    static void maxpool2d(
        const T* input,
        T* output,
        int N,
        int C,
        int H_in,
        int W_in,
        int H_out,
        int W_out,
        int kernel_h,
        int kernel_w,
        int stride_h,
        int stride_w,
        int padding_h,
        int padding_w,
        KernelBackend backend = KernelBackend::AUTO
    ) {
        // Ensure kernels are initialized
        KernelRegistryInitializer::initialize();
        
        MaxPool2DFunc<T> func = nullptr;
        
        // Get kernel from registry
        if (backend == KernelBackend::AUTO) {
            func = MaxPool2DRegistry<T>::instance().get_best_kernel();
        } else {
            func = MaxPool2DRegistry<T>::instance().get_kernel(backend);
        }
        
        if (func) {
            func(input, output, N, C, H_in, W_in, H_out, W_out,
                 kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
        } else {
            throw std::runtime_error("No MaxPool2D kernel available for requested backend");
        }
    }
    
    /**
     * @brief AvgPool2D operation
     * 
     * TensorRT-style: Automatically dispatches to best available implementation.
     * Average pooling excludes padding (count_include_pad=false).
     * 
     * @param backend Backend selection:
     *   - AUTO (default): Auto-select best available
     *   - CPU/CPU_AVX2/etc: Force specific backend
     */
    template<typename T>
    static void avgpool2d(
        const T* input,
        T* output,
        int N,
        int C,
        int H_in,
        int W_in,
        int H_out,
        int W_out,
        int kernel_h,
        int kernel_w,
        int stride_h,
        int stride_w,
        int padding_h,
        int padding_w,
        KernelBackend backend = KernelBackend::AUTO
    ) {
        // Ensure kernels are initialized
        KernelRegistryInitializer::initialize();
        
        AvgPool2DFunc<T> func = nullptr;
        
        // Get kernel from registry
        if (backend == KernelBackend::AUTO) {
            func = AvgPool2DRegistry<T>::instance().get_best_kernel();
        } else {
            func = AvgPool2DRegistry<T>::instance().get_kernel(backend);
        }
        
        if (func) {
            func(input, output, N, C, H_in, W_in, H_out, W_out,
                 kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
        } else {
            throw std::runtime_error("No AvgPool2D kernel available for requested backend");
        }
    }
    
    /**
     * @brief Get the best available backend for MaxPool2D
     */
    DEFINE_BEST_BACKEND_GETTER(get_best_backend_maxpool, MaxPool2DRegistry)
    
    /**
     * @brief Get the best available backend for AvgPool2D
     */
    DEFINE_BEST_BACKEND_GETTER(get_best_backend_avgpool, AvgPool2DRegistry)
    
    /**
     * @brief Check if specific backend is available for MaxPool2D
     */
    DEFINE_BACKEND_CHECKER(is_backend_available_maxpool, MaxPool2DRegistry)
    
    /**
     * @brief Check if specific backend is available for AvgPool2D
     */
    DEFINE_BACKEND_CHECKER(is_backend_available_avgpool, AvgPool2DRegistry)
};

} // namespace kernels
} // namespace mini_infer
