#include "mini_infer/kernels/gemm.h"
#include <cstring>
#include <cstdint>
#include <stdexcept>

namespace mini_infer {
namespace kernels {

/**
 * @brief CPU implementation of GEMM kernels
 * 
 * TensorRT-style: Auto-register at program startup
 * 
 * Future optimizations:
 * - AVX2/AVX512 vectorization
 * - OpenMP parallelization
 * - Cache blocking
 * - OpenBLAS/MKL integration
 */

namespace cpu {

// GEMM NN: C = A @ B
template<typename T>
void gemm_nn_impl(
    const T* A,
    const T* B,
    T* C,
    int M,
    int N,
    int K) {
    
    std::memset(C, 0, sizeof(T) * M * N);
    
    // Optimized loop order: M -> K -> N
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            T a_val = A[m * K + k];
            const T* b_row = B + k * N;
            T* c_row = C + m * N;
            
            // Vectorizable inner loop
            for (int n = 0; n < N; ++n) {
                c_row[n] += a_val * b_row[n];
            }
        }
    }
}

// GEMM NT: C = A @ B^T
template<typename T>
void gemm_nt_impl(
    const T* A,
    const T* B,
    T* C,
    int M,
    int N,
    int K) {
    
    std::memset(C, 0, sizeof(T) * M * N);
    
    for (int m = 0; m < M; ++m) {
        const T* a_row = A + m * K;
        T* c_row = C + m * N;
        
        for (int n = 0; n < N; ++n) {
            const T* b_row = B + n * K;
            T sum = 0;
            
            // Loop unrolling for better performance
            int k = 0;
            for (; k + 3 < K; k += 4) {
                sum += a_row[k + 0] * b_row[k + 0];
                sum += a_row[k + 1] * b_row[k + 1];
                sum += a_row[k + 2] * b_row[k + 2];
                sum += a_row[k + 3] * b_row[k + 3];
            }
            
            for (; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
            
            c_row[n] = sum;
        }
    }
}

// ============================================================================
// Explicit Registration Function 
// ============================================================================

void register_gemm_kernels() {
    // CPU availability checker (inline lambda)
    auto is_cpu_available = []() { return true; };
    // Register CPU GEMM_NN implementations
    GEMMRegistry_NN<float>::instance().register_kernel(
        KernelBackend::CPU,
        gemm_nn_impl<float>,
        is_cpu_available,
        100  // Priority: CPU is baseline
    );
    
    GEMMRegistry_NN<int32_t>::instance().register_kernel(
        KernelBackend::CPU,
        gemm_nn_impl<int32_t>,
        is_cpu_available,
        100
    );
    
    // Register CPU GEMM_NT implementations
    GEMMRegistry_NT<float>::instance().register_kernel(
        KernelBackend::CPU,
        gemm_nt_impl<float>,
        is_cpu_available,
        100
    );
    
    GEMMRegistry_NT<int32_t>::instance().register_kernel(
        KernelBackend::CPU,
        gemm_nt_impl<int32_t>,
        is_cpu_available,
        100
    );
}

} // namespace cpu

} // namespace kernels
} // namespace mini_infer
