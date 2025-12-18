#pragma once

#include "mini_infer/kernels/kernel_types.h"
#include "mini_infer/kernels/kernel_base.h"
#include "mini_infer/kernels/kernel_registry_template.h"
#include <stdexcept>

namespace mini_infer {
namespace kernels {

/**
 * @brief Bias function signature
 */
template<typename T>
using BiasFunc = void(*)(T*, const T*, int, int, int);

/**
 * @brief Bias Registry
 * 
 * Using template-based registry to eliminate code duplication.
 */
DEFINE_REGISTRY_ALIAS(BiasRegistry, BiasFunc);

/**
 * @brief Bias Kernel Interface
 * 
 * TensorRT-style: Auto-dispatch to best available implementation.
 * Provides optimized bias addition operations commonly used in
 * neural network layers (Conv2D, Linear, BatchNorm, etc.)
 */
class BiasKernel {
public:
    /**
     * @brief Add channel-wise bias to tensor
     * 
     * TensorRT-style: Automatically dispatches to best available implementation.
     * Performs: output[b, c, s] += bias[c] for all b, c, s
     * 
     * @tparam T Data type (float, int32_t, etc.)
     * @param output Output tensor data [batch_size, channels, spatial_size]
     * @param bias Bias data [channels]
     * @param batch_size Number of batches
     * @param channels Number of channels
     * @param spatial_size Spatial dimension size (H*W for Conv2D, 1 for Linear)
     * @param backend Backend selection:
     *   - AUTO (default): Auto-select best available
     *   - CPU/CPU_AVX2/etc: Force specific backend
     * 
     * Layout examples:
     * - Conv2D: [N, C, H*W] where spatial_size = H*W
     * - Linear: [batch, features] where spatial_size = 1
     */
    template<typename T>
    static void add_channel_bias(
        T* output,
        const T* bias,
        int batch_size,
        int channels,
        int spatial_size,
        KernelBackend backend = KernelBackend::AUTO
    ) {
        BiasFunc<T> func = nullptr;
        
        // Get kernel from registry
        if (backend == KernelBackend::AUTO) {
            // Auto-select best available
            func = BiasRegistry<T>::instance().get_best_kernel();
        } else {
            // Use specific backend
            func = BiasRegistry<T>::instance().get_kernel(backend);
        }
        
        if (func) {
            func(output, bias, batch_size, channels, spatial_size);
        } else {
            throw std::runtime_error("No Bias kernel available for requested backend");
        }
    }
    
    /**
     * @brief Get the best available backend
     */
    DEFINE_BEST_BACKEND_GETTER(get_best_backend, BiasRegistry)
    
    /**
     * @brief Check if specific backend is available
     */
    DEFINE_BACKEND_CHECKER(is_backend_available, BiasRegistry)
};

} // namespace kernels
} // namespace mini_infer
