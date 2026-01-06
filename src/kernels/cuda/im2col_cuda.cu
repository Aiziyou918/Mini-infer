#include "mini_infer/kernels/im2col.h"
#include "mini_infer/kernels/kernel_base.h"
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/utils/logger.h"

#include <string>

namespace mini_infer {
namespace kernels {
namespace cuda {

/**
 * @brief Im2Col CUDA kernel
 *
 * Transforms image patches into column format for convolution.
 * Each thread handles one output element in the col_buffer.
 *
 * @param input Input data [C, H, W]
 * @param col_buffer Output column buffer [C*kernel_h*kernel_w, out_height*out_width]
 * @param channels Number of input channels
 * @param height Input height
 * @param width Input width
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param stride_h Stride in height
 * @param stride_w Stride in width
 * @param padding_h Padding in height
 * @param padding_w Padding in width
 * @param dilation_h Dilation in height
 * @param dilation_w Dilation in width
 * @param out_height Output height
 * @param out_width Output width
 */
__global__ void im2col_kernel(
    const float* __restrict__ input,
    float* __restrict__ col_buffer,
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
    int out_width
) {
    // Total number of output elements
    int col_height = channels * kernel_h * kernel_w;
    int col_width = out_height * out_width;
    int total_elements = col_height * col_width;

    // Grid-stride loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        // Decode index: i = col_row * col_width + col_col
        int col_row = i / col_width;
        int col_col = i % col_width;

        // Decode col_row: col_row = c * kernel_h * kernel_w + kh * kernel_w + kw
        int c = col_row / (kernel_h * kernel_w);
        int kernel_idx = col_row % (kernel_h * kernel_w);
        int kh = kernel_idx / kernel_w;
        int kw = kernel_idx % kernel_w;

        // Decode col_col: col_col = oh * out_width + ow
        int oh = col_col / out_width;
        int ow = col_col % out_width;

        // Calculate input position
        int input_row = -padding_h + kh * dilation_h + oh * stride_h;
        int input_col = -padding_w + kw * dilation_w + ow * stride_w;

        // Check bounds and write to col_buffer
        if (input_row >= 0 && input_row < height &&
            input_col >= 0 && input_col < width) {
            int input_idx = c * height * width + input_row * width + input_col;
            col_buffer[i] = input[input_idx];
        } else {
            col_buffer[i] = 0.0f;
        }
    }
}

/**
 * @brief Im2Col CUDA implementation
 */
void im2col_cuda_impl(
    const float* input,
    float* col_buffer,
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
    int out_width
) {
    int col_height = channels * kernel_h * kernel_w;
    int col_width = out_height * out_width;
    int total_elements = col_height * col_width;

    const int threads = 256;
    int blocks = std::min((total_elements + threads - 1) / threads, 65535);

    // Get current CUDA stream
    cudaStream_t stream = nullptr;
    auto* ctx = get_current_device_context();
    if (ctx) {
        auto* cuda_ctx = static_cast<backends::cuda::CUDADeviceContext*>(ctx);
        stream = cuda_ctx->stream();
    }

    im2col_kernel<<<blocks, threads, 0, stream>>>(
        input, col_buffer, channels, height, width,
        kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w, dilation_h, dilation_w,
        out_height, out_width
    );

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDA] Im2Col kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

/**
 * @brief Check if CUDA is available
 */
bool is_cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

/**
 * @brief Im2Col CUDA kernel registrar
 */
namespace {
    void register_im2col_cuda_kernels() {
        Im2ColRegistry<float>::instance().register_kernel(
            KernelBackend::CUDA,
            im2col_cuda_impl,
            is_cuda_available,
            200  // Higher priority than CPU
        );
    }

    struct Im2ColCUDARegistrar {
        Im2ColCUDARegistrar() {
            register_im2col_cuda_kernels();
        }
    };
    static Im2ColCUDARegistrar g_im2col_cuda_registrar;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace mini_infer
