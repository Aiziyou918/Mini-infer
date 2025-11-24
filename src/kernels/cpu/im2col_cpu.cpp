#include "mini_infer/kernels/im2col.h"
#include <stdexcept>

namespace mini_infer {
namespace kernels {

/**
 * @brief CPU implementation of Im2Col
 * 
 * TensorRT-style: Auto-register at program startup
 * Reference: Caffe's im2col
 */

namespace cpu {

template<typename T>
void im2col_impl(
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
    int out_width) {
    
    int channel_size = height * width;
    
    for (int c = 0; c < channels; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Calculate input row and column
                int input_row_start = -padding_h + kh * dilation_h;
                int input_col_start = -padding_w + kw * dilation_w;
                
                // Column index
                int col_idx = (c * kernel_h * kernel_w + kh * kernel_w + kw);
                
                for (int oh = 0; oh < out_height; ++oh) {
                    for (int ow = 0; ow < out_width; ++ow) {
                        // Calculate corresponding input position
                        int input_row = input_row_start + oh * stride_h;
                        int input_col = input_col_start + ow * stride_w;
                        
                        int col_buffer_idx = col_idx * out_height * out_width + oh * out_width + ow;
                        
                        // Check if position is within valid input bounds
                        if (input_row >= 0 && input_row < height &&
                            input_col >= 0 && input_col < width) {
                            int input_idx = c * channel_size + input_row * width + input_col;
                            col_buffer[col_buffer_idx] = input[input_idx];
                        } else {
                            // Padding area
                            col_buffer[col_buffer_idx] = T(0);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Explicit Registration Function 
// ============================================================================

void register_im2col_kernels() {
    // CPU availability checker (inline lambda)
    auto is_cpu_available = []() { return true; };
    
    // Register CPU Im2Col implementation
    Im2ColRegistry<float>::instance().register_kernel(
        KernelBackend::CPU,
        im2col_impl<float>,
        is_cpu_available,
        100  // Priority: CPU is baseline
    );
}

} // namespace cpu

} // namespace kernels
} // namespace mini_infer
