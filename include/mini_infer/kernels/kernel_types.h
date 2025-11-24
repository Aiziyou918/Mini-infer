#pragma once

namespace mini_infer {
namespace kernels {

/**
 * @brief Backend type for kernel dispatch
 * 
 * Defines available computational backends for kernel execution.
 * Kernels will automatically dispatch to the best available backend.
 */
enum class KernelBackend {
    AUTO,          // Auto-select best available backend
    CPU,           // Basic CPU implementation
    CPU_AVX2,      // AVX2 vectorized implementation
    CPU_AVX512,    // AVX512 vectorized implementation
    CPU_BLAS,      // OpenBLAS/MKL implementation
    CUDA,          // Basic CUDA implementation
    CUDA_CUBLAS    // cuBLAS optimized implementation
};

/**
 * @brief Kernel utilities
 */
class KernelUtils {
public:
    /**
     * @brief Check if specific backend is available on current hardware
     * 
     * @param backend Backend to check
     * @return true if available, false otherwise
     */
    static bool is_backend_available(KernelBackend backend);
    
    /**
     * @brief Get best available backend for current hardware
     * 
     * Checks available backends in order of preference:
     * CUDA_CUBLAS > CUDA > CPU_BLAS > CPU_AVX512 > CPU_AVX2 > CPU
     * 
     * @return Best available backend
     */
    static KernelBackend get_best_backend();
    
    /**
     * @brief Get backend name string
     * 
     * @param backend Backend type
     * @return Human-readable backend name
     */
    static const char* backend_name(KernelBackend backend);
};

} // namespace kernels
} // namespace mini_infer
