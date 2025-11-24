#include "mini_infer/kernels/pooling.h"
#include <algorithm>
#include <limits>
#include <cmath>

namespace mini_infer {
namespace kernels {

/**
 * @brief CPU implementation of Pooling operations
 * 
 * TensorRT-style: Auto-register at program startup
 */

namespace cpu {

/**
 * @brief CPU MaxPool2D implementation
 * 
 * Takes maximum value in each pooling window.
 * Handles padding by treating padded areas as -infinity.
 */
template<typename T>
void maxpool2d_impl(
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
    int padding_w) {
    
    // Process each batch and channel independently
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const T* input_nc = input + (n * C + c) * H_in * W_in;
            T* output_nc = output + (n * C + c) * H_out * W_out;
            
            // Iterate over output spatial dimensions
            for (int h_out = 0; h_out < H_out; ++h_out) {
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    // Calculate input window boundaries
                    int h_start = h_out * stride_h - padding_h;
                    int w_start = w_out * stride_w - padding_w;
                    int h_end = std::min(h_start + kernel_h, H_in);
                    int w_end = std::min(w_start + kernel_w, W_in);
                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);
                    
                    // Find maximum in window
                    T max_val = std::numeric_limits<T>::lowest();
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            T val = input_nc[h * W_in + w];
                            max_val = std::max(max_val, val);
                        }
                    }
                    
                    output_nc[h_out * W_out + w_out] = max_val;
                }
            }
        }
    }
}

/**
 * @brief CPU AvgPool2D implementation
 * 
 * Takes average value in each pooling window.
 * TensorRT-style: Excludes padding from average calculation (count_include_pad=false).
 */
template<typename T>
void avgpool2d_impl(
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
    int padding_w) {
    
    // Process each batch and channel independently
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const T* input_nc = input + (n * C + c) * H_in * W_in;
            T* output_nc = output + (n * C + c) * H_out * W_out;
            
            // Iterate over output spatial dimensions
            for (int h_out = 0; h_out < H_out; ++h_out) {
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    // Calculate input window boundaries
                    int h_start = h_out * stride_h - padding_h;
                    int w_start = w_out * stride_w - padding_w;
                    int h_end = std::min(h_start + kernel_h, H_in);
                    int w_end = std::min(w_start + kernel_w, W_in);
                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);
                    
                    // Calculate sum and count (excluding padding)
                    T sum = 0;
                    int count = 0;
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            sum += input_nc[h * W_in + w];
                            ++count;
                        }
                    }
                    
                    // Average (excluding padding)
                    output_nc[h_out * W_out + w_out] = (count > 0) ? (sum / static_cast<T>(count)) : T(0);
                }
            }
        }
    }
}

} // namespace cpu

// ============================================================================
// Explicit Registration Function 
// ============================================================================

void register_pooling_kernels() {
    // CPU availability checker (inline lambda)
    auto is_cpu_available = []() { return true; };
    
    // Register MaxPool2D kernels
    MaxPool2DRegistry<float>::instance().register_kernel(
        KernelBackend::CPU,
        cpu::maxpool2d_impl<float>,
        is_cpu_available
    );
    
    MaxPool2DRegistry<int32_t>::instance().register_kernel(
        KernelBackend::CPU,
        cpu::maxpool2d_impl<int32_t>,
        is_cpu_available
    );
    
    // Register AvgPool2D kernels
    AvgPool2DRegistry<float>::instance().register_kernel(
        KernelBackend::CPU,
        cpu::avgpool2d_impl<float>,
        is_cpu_available
    );
    
    AvgPool2DRegistry<int32_t>::instance().register_kernel(
        KernelBackend::CPU,
        cpu::avgpool2d_impl<int32_t>,
        is_cpu_available
    );
}

} // namespace kernels
} // namespace mini_infer
