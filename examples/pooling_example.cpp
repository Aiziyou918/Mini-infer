#include "mini_infer/operators/pooling.h"
#include "mini_infer/core/tensor.h"
#include <iostream>
#include <iomanip>

using namespace mini_infer;

void print_tensor(const char* name, const core::Tensor& tensor) {
    std::cout << name << " shape: [";
    for (size_t i = 0; i < tensor.shape().ndim(); ++i) {
        std::cout << tensor.shape()[i];
        if (i < tensor.shape().ndim() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Pooling Layer Example\n";
    std::cout << "========================================\n\n";
    
    // Create input tensor [1, 2, 4, 4]
    auto input = std::make_shared<core::Tensor>(
        core::Shape({1, 2, 4, 4}),
        core::DataType::FLOAT32
    );
    
    // Fill with test data
    float* input_data = static_cast<float*>(input->data());
    
    // Channel 0: sequential values 0-15
    for (int i = 0; i < 16; ++i) {
        input_data[i] = static_cast<float>(i);
    }
    
    // Channel 1: values 16-31
    for (int i = 0; i < 16; ++i) {
        input_data[16 + i] = static_cast<float>(16 + i);
    }
    
    std::cout << "Input tensor:\n";
    print_tensor("Input", *input);
    std::cout << "\nChannel 0:\n";
    std::cout << "[[ 0,  1,  2,  3],\n";
    std::cout << " [ 4,  5,  6,  7],\n";
    std::cout << " [ 8,  9, 10, 11],\n";
    std::cout << " [12, 13, 14, 15]]\n";
    std::cout << "\nChannel 1:\n";
    std::cout << "[[16, 17, 18, 19],\n";
    std::cout << " [20, 21, 22, 23],\n";
    std::cout << " [24, 25, 26, 27],\n";
    std::cout << " [28, 29, 30, 31]]\n\n";
    
    // ========================================================================
    // Example 1: MaxPool2D with kernel=2, stride=2
    // ========================================================================
    
    std::cout << "========================================\n";
    std::cout << "Example 1: MaxPool2D (2x2, stride=2)\n";
    std::cout << "========================================\n";
    
    operators::PoolingParam maxpool_param(
        operators::PoolingType::MAX,
        2, 2,  // kernel_h, kernel_w
        2, 2   // stride_h, stride_w
    );
    
    operators::Pooling maxpool_op(maxpool_param);
    
    // Infer output shape
    std::vector<core::Shape> input_shapes = {input->shape()};
    std::vector<core::Shape> output_shapes;
    auto status = maxpool_op.infer_shape(input_shapes, output_shapes);
    
    if (status != core::Status::SUCCESS) {
        std::cerr << "Failed to infer shape for MaxPool\n";
        return 1;
    }
    
    // Create output tensor
    auto maxpool_output = std::make_shared<core::Tensor>(
        output_shapes[0],
        core::DataType::FLOAT32
    );
    
    // Forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {maxpool_output};
    
    status = maxpool_op.forward(inputs, outputs);
    
    if (status != core::Status::SUCCESS) {
        std::cerr << "MaxPool forward failed\n";
        return 1;
    }
    
    std::cout << "\nOutput:\n";
    print_tensor("MaxPool Output", *maxpool_output);
    
    const float* maxpool_data = static_cast<const float*>(maxpool_output->data());
    std::cout << "\nChannel 0 (expected: [5, 7, 13, 15]):\n";
    std::cout << "[[" << maxpool_data[0] << ", " << maxpool_data[1] << "],\n";
    std::cout << " [" << maxpool_data[2] << ", " << maxpool_data[3] << "]]\n";
    std::cout << "\nChannel 1 (expected: [21, 23, 29, 31]):\n";
    std::cout << "[[" << maxpool_data[4] << ", " << maxpool_data[5] << "],\n";
    std::cout << " [" << maxpool_data[6] << ", " << maxpool_data[7] << "]]\n\n";
    
    // ========================================================================
    // Example 2: AvgPool2D with kernel=2, stride=2
    // ========================================================================
    
    std::cout << "========================================\n";
    std::cout << "Example 2: AvgPool2D (2x2, stride=2)\n";
    std::cout << "========================================\n";
    
    operators::PoolingParam avgpool_param(
        operators::PoolingType::AVERAGE,
        2, 2,  // kernel_h, kernel_w
        2, 2   // stride_h, stride_w
    );
    
    operators::Pooling avgpool_op(avgpool_param);
    
    // Create output tensor
    auto avgpool_output = std::make_shared<core::Tensor>(
        output_shapes[0],
        core::DataType::FLOAT32
    );
    
    // Forward pass
    outputs[0] = avgpool_output;
    status = avgpool_op.forward(inputs, outputs);
    
    if (status != core::Status::SUCCESS) {
        std::cerr << "AvgPool forward failed\n";
        return 1;
    }
    
    std::cout << "\nOutput:\n";
    print_tensor("AvgPool Output", *avgpool_output);
    
    const float* avgpool_data = static_cast<const float*>(avgpool_output->data());
    std::cout << "\nChannel 0 (expected: [2.5, 4.5, 10.5, 12.5]):\n";
    std::cout << "[[" << avgpool_data[0] << ", " << avgpool_data[1] << "],\n";
    std::cout << " [" << avgpool_data[2] << ", " << avgpool_data[3] << "]]\n";
    std::cout << "\nChannel 1 (expected: [18.5, 20.5, 26.5, 28.5]):\n";
    std::cout << "[[" << avgpool_data[4] << ", " << avgpool_data[5] << "],\n";
    std::cout << " [" << avgpool_data[6] << ", " << avgpool_data[7] << "]]\n\n";
    
    // ========================================================================
    // Example 3: MaxPool with padding
    // ========================================================================
    
    std::cout << "========================================\n";
    std::cout << "Example 3: MaxPool (3x3, stride=2, pad=1)\n";
    std::cout << "========================================\n";
    
    operators::PoolingParam maxpool_pad_param(
        operators::PoolingType::MAX,
        3, 3,  // kernel_h, kernel_w
        2, 2,  // stride_h, stride_w
        1, 1   // padding_h, padding_w
    );
    
    operators::Pooling maxpool_pad_op(maxpool_pad_param);
    
    // Infer output shape
    maxpool_pad_op.infer_shape(input_shapes, output_shapes);
    
    auto maxpool_pad_output = std::make_shared<core::Tensor>(
        output_shapes[0],
        core::DataType::FLOAT32
    );
    
    outputs[0] = maxpool_pad_output;
    status = maxpool_pad_op.forward(inputs, outputs);
    
    if (status != core::Status::SUCCESS) {
        std::cerr << "MaxPool with padding forward failed\n";
        return 1;
    }
    
    std::cout << "\nOutput:\n";
    print_tensor("MaxPool (padded) Output", *maxpool_pad_output);
    std::cout << "\nOutput shape: [1, 2, 2, 2] (with padding)\n\n";
    
    // ========================================================================
    // Example 4: Global Average Pooling (kernel=4x4)
    // ========================================================================
    
    std::cout << "========================================\n";
    std::cout << "Example 4: Global Average Pooling\n";
    std::cout << "========================================\n";
    
    operators::PoolingParam global_avg_param(
        operators::PoolingType::AVERAGE,
        4, 4,  // kernel_h, kernel_w (full spatial size)
        1, 1   // stride_h, stride_w
    );
    
    operators::Pooling global_avg_op(global_avg_param);
    
    // Infer output shape
    global_avg_op.infer_shape(input_shapes, output_shapes);
    
    auto global_avg_output = std::make_shared<core::Tensor>(
        output_shapes[0],
        core::DataType::FLOAT32
    );
    
    outputs[0] = global_avg_output;
    status = global_avg_op.forward(inputs, outputs);
    
    if (status != core::Status::SUCCESS) {
        std::cerr << "Global average pooling forward failed\n";
        return 1;
    }
    
    std::cout << "\nOutput:\n";
    print_tensor("Global AvgPool Output", *global_avg_output);
    
    const float* global_avg_data = static_cast<const float*>(global_avg_output->data());
    std::cout << "\nChannel 0 average: " << global_avg_data[0] << " (expected: 7.5)\n";
    std::cout << "Channel 1 average: " << global_avg_data[1] << " (expected: 23.5)\n\n";
    
    std::cout << "========================================\n";
    std::cout << "All examples completed successfully!\n";
    std::cout << "========================================\n";
    
    return 0;
}
