#pragma once

#include "mini_infer/kernels/kernel_types.h"
#include "mini_infer/kernels/kernel_base.h"
#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/kernels/kernel_registry_template.h"
#include <stdexcept>

namespace mini_infer {
namespace kernels {

/**
 * @brief GEMM Kernel Interface
 * 
 * TensorRT-style kernel registry and dispatch system.
 * Backend implementations auto-register themselves at startup.
 * 
 * Supports different matrix multiplication modes:
 * - NN: C = A @ B (both normal)
 * - NT: C = A @ B^T (B transposed)
 */

/**
 * @brief GEMM function signatures
 */
template<typename T>
using GEMMFunc_NN = void(*)(const T*, const T*, T*, int, int, int);

template<typename T>
using GEMMFunc_NT = void(*)(const T*, const T*, T*, int, int, int);

/**
 * @brief GEMM Registry for NN operation (C = A @ B)
 * 
 * Using template-based registry to eliminate code duplication.
 */
DEFINE_REGISTRY_ALIAS(GEMMRegistry_NN, GEMMFunc_NN);

/**
 * @brief GEMM Registry for NT operation (C = A @ B^T)
 * 
 * Using template-based registry to eliminate code duplication.
 */
DEFINE_REGISTRY_ALIAS(GEMMRegistry_NT, GEMMFunc_NT);

/**
 * @brief GEMM Kernel dispatcher
 * 
 * TensorRT-style: Uses registry to automatically select best implementation.
 */
class GEMMKernel {
public:
    
    /**
     * @brief Optimized GEMM: C = A @ B (No transpose)
     * 
     * TensorRT-style: Automatically dispatches to best available implementation.
     * Implementations are registered at program startup via AutoRegister.
     * 
     * @param backend Backend selection:
     *   - AUTO (default): Auto-select best available
     *   - CPU/CPU_AVX2/etc: Force specific backend
     */
    template<typename T>
    static void gemm_nn(
        const T* A,
        const T* B,
        T* C,
        int M,
        int N,
        int K,
        KernelBackend backend = KernelBackend::AUTO
    ) {
        // Ensure kernels are initialized
        KernelRegistryInitializer::initialize();
        
        GEMMFunc_NN<T> func = nullptr;
        
        // Get kernel from registry
        if (backend == KernelBackend::AUTO) {
            // Auto-select best available
            func = GEMMRegistry_NN<T>::instance().get_best_kernel();
        } else {
            // Use specific backend
            func = GEMMRegistry_NN<T>::instance().get_kernel(backend);
        }
        
        if (func) {
            func(A, B, C, M, N, K);
        } else {
            throw std::runtime_error("No GEMM_NN kernel available for requested backend");
        }
    }
    
    /**
     * @brief Optimized GEMM: C = A @ B^T (Transpose B)
     * 
     * TensorRT-style: Automatically dispatches to best available implementation.
     * 
     * @param backend Backend selection:
     *   - AUTO (default): Auto-select best available
     *   - CPU/CPU_AVX2/etc: Force specific backend
     */
    template<typename T>
    static void gemm_nt(
        const T* A,
        const T* B,
        T* C,
        int M,
        int N,
        int K,
        KernelBackend backend = KernelBackend::AUTO
    ) {
        // Ensure kernels are initialized
        KernelRegistryInitializer::initialize();
        
        GEMMFunc_NT<T> func = nullptr;
        
        // Get kernel from registry
        if (backend == KernelBackend::AUTO) {
            // Auto-select best available
            func = GEMMRegistry_NT<T>::instance().get_best_kernel();
        } else {
            // Use specific backend
            func = GEMMRegistry_NT<T>::instance().get_kernel(backend);
        }
        
        if (func) {
            func(A, B, C, M, N, K);
        } else {
            throw std::runtime_error("No GEMM_NT kernel available for requested backend");
        }
    }
    
    /**
     * @brief Get the best available backend for GEMM_NN
     */
    DEFINE_BEST_BACKEND_GETTER(get_best_backend_nn, GEMMRegistry_NN)
    
    /**
     * @brief Get the best available backend for GEMM_NT
     */
    DEFINE_BEST_BACKEND_GETTER(get_best_backend_nt, GEMMRegistry_NT)
    
    /**
     * @brief Check if specific backend is available for GEMM_NN
     */
    DEFINE_BACKEND_CHECKER(is_backend_available_nn, GEMMRegistry_NN)
    
    /**
     * @brief Check if specific backend is available for GEMM_NT
     */
    DEFINE_BACKEND_CHECKER(is_backend_available_nt, GEMMRegistry_NT)
};

} // namespace kernels
} // namespace mini_infer
