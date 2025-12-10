#include "mini_infer/operators/conv2d.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "mini_infer/core/buffer.h"
#include "mini_infer/core/op_type.h"
#include "mini_infer/kernels/bias.h"
#include "mini_infer/kernels/gemm.h"
#include "mini_infer/kernels/im2col.h"

namespace mini_infer {
namespace operators {

Conv2D::Conv2D(const Conv2DParam& param) : Operator(core::op_names::kConv), param_(param) {}

core::Status Conv2D::forward(const std::vector<std::shared_ptr<core::Tensor>>& inputs,
                             std::vector<std::shared_ptr<core::Tensor>>& outputs) {
    // Input validation
    size_t expected_inputs = param_.use_bias ? 3 : 2;
    if (inputs.size() != expected_inputs) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    const auto& input = inputs[0];   // [N, C_in, H_in, W_in]
    const auto& weight = inputs[1];  // [C_out, C_in/groups, kernel_h, kernel_w]

    if (!input || !weight) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Validate bias if needed
    if (param_.use_bias) {
        const auto& bias = inputs[2];
        if (!bias) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }

    // Get input shapes
    const auto& input_shape = input->shape();
    const auto& weight_shape = weight->shape();

    // Validate shapes
    if (input_shape.ndim() != 4 || weight_shape.ndim() != 4) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Extract dimensions
    int N = static_cast<int>(input_shape[0]);
    int C_in = static_cast<int>(input_shape[1]);
    int H_in = static_cast<int>(input_shape[2]);
    int W_in = static_cast<int>(input_shape[3]);

    int C_out = static_cast<int>(weight_shape[0]);
    int C_in_per_group = static_cast<int>(weight_shape[1]);
    int kernel_h = static_cast<int>(weight_shape[2]);
    int kernel_w = static_cast<int>(weight_shape[3]);

    // Validate parameters
    // TODO: Groups convolution not yet implemented, only groups=1 supported
    if (param_.groups != 1) {
        return core::Status::ERROR_NOT_IMPLEMENTED;
    }
    if (C_in != C_in_per_group * param_.groups) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Calculate output dimensions
    int H_out =
        (H_in + 2 * param_.padding_h - param_.dilation_h * (kernel_h - 1) - 1) / param_.stride_h +
        1;
    int W_out =
        (W_in + 2 * param_.padding_w - param_.dilation_w * (kernel_w - 1) - 1) / param_.stride_w +
        1;

    // Create output tensor
    std::vector<int64_t> output_dims = {static_cast<int64_t>(N), static_cast<int64_t>(C_out),
                                        static_cast<int64_t>(H_out), static_cast<int64_t>(W_out)};
    core::Shape output_shape(output_dims);

    auto output = core::Tensor::create(output_shape, input->dtype());
    if (!output) {
        return core::Status::ERROR_OUT_OF_MEMORY;
    }

    // Perform computation based on data type
    const auto dtype = input->dtype();

    if (dtype == core::DataType::FLOAT32) {
        const float* input_data = static_cast<const float*>(input->data());
        const float* weight_data = static_cast<const float*>(weight->data());
        float* output_data = static_cast<float*>(output->data());

        // CPU-optimized: Per-batch GEMM for better cache utilization
        // Directly produces NCHW layout without transpose overhead
        int spatial_size = H_out * W_out;
        int col_size_per_batch = C_in * kernel_h * kernel_w * spatial_size;

        // Allocate col_buffer for single batch (better cache locality)
        core::Buffer<float> col_buffer(col_size_per_batch);

        // GEMM parameters
        int M = C_out;
        int N_gemm = spatial_size;
        int K = C_in * kernel_h * kernel_w;

        // Process each batch separately
        for (int n = 0; n < N; ++n) {
            const float* input_n = input_data + n * C_in * H_in * W_in;
            float* output_n = output_data + n * C_out * spatial_size;

            // Step 1: im2col for this batch
            kernels::Im2ColKernel::im2col<float>(
                input_n, col_buffer.data(), C_in, H_in, W_in, kernel_h, kernel_w, param_.stride_h,
                param_.stride_w, param_.padding_h, param_.padding_w, param_.dilation_h,
                param_.dilation_w, H_out, W_out);

            // Step 2: GEMM for this batch
            // weight: [C_out, K]
            // col_buffer: [K, spatial_size]
            // output_n: [C_out, spatial_size] (NCHW layout, naturally)
            kernels::GEMMKernel::gemm_nn<float>(weight_data, col_buffer.data(), output_n, M, N_gemm,
                                                K);
        }

        // Add bias if needed
        if (param_.use_bias) {
            const float* bias_data = static_cast<const float*>(inputs[2]->data());
            kernels::BiasKernel::add_channel_bias<float>(output_data, bias_data,
                                                         N,             // batch_size
                                                         C_out,         // channels
                                                         H_out * W_out  // spatial_size
            );
        }

        // TensorRT-style: Apply inline activation if set
        // This enables automatic kernel fusion without explicit fusion pass
        if (param_.activation.is_enabled()) {
            int total_elements = N * C_out * H_out * W_out;
            for (int i = 0; i < total_elements; ++i) {
                output_data[i] = apply_activation(output_data[i], param_.activation);
            }
        }

    } else {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Set output
    outputs.clear();
    outputs.push_back(output);

    return core::Status::SUCCESS;
}

core::Status Conv2D::infer_shape(const std::vector<core::Shape>& input_shapes,
                                 std::vector<core::Shape>& output_shapes) {
    // Validate input count
    size_t expected_inputs = param_.use_bias ? 3 : 2;
    if (input_shapes.size() != expected_inputs) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    const auto& input_shape = input_shapes[0];   // [N, C_in, H_in, W_in]
    const auto& weight_shape = input_shapes[1];  // [C_out, C_in/groups, kernel_h, kernel_w]

    // Validate shapes
    if (input_shape.ndim() != 4 || weight_shape.ndim() != 4) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Extract dimensions
    int64_t N = input_shape[0];
    int64_t C_in = input_shape[1];
    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];

    int64_t C_out = weight_shape[0];
    int64_t C_in_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Validate channel dimensions
    // TODO: Groups convolution not yet implemented, only groups=1 supported
    if (param_.groups != 1) {
        return core::Status::ERROR_NOT_IMPLEMENTED;
    }
    if (C_in != C_in_per_group * param_.groups) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Validate bias shape if needed
    if (param_.use_bias && input_shapes.size() > 2) {
        const auto& bias_shape = input_shapes[2];
        if (bias_shape.ndim() != 1 || bias_shape[0] != C_out) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }

    // Calculate output dimensions
    int64_t H_out =
        (H_in + 2 * param_.padding_h - param_.dilation_h * (kernel_h - 1) - 1) / param_.stride_h +
        1;
    int64_t W_out =
        (W_in + 2 * param_.padding_w - param_.dilation_w * (kernel_w - 1) - 1) / param_.stride_w +
        1;

    output_shapes.clear();
    output_shapes.push_back(core::Shape({N, C_out, H_out, W_out}));

    return core::Status::SUCCESS;
}

// Register Conv2D operator
REGISTER_OPERATOR(Conv2D, Conv2D);

}  // namespace operators
}  // namespace mini_infer
