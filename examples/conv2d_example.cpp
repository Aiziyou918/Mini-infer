#include "mini_infer/mini_infer.h"
#include <iostream>
#include <iomanip>
#include <random>

using namespace mini_infer;

void print_tensor_info(const std::shared_ptr<core::Tensor>& tensor, const std::string& name) {
    std::cout << name << " shape: [";
    for (size_t i = 0; i < tensor->shape().ndim(); ++i) {
        std::cout << tensor->shape()[i];
        if (i < tensor->shape().ndim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void print_output_sample(const std::shared_ptr<core::Tensor>& tensor, int samples = 5) {
    const float* data = static_cast<const float*>(tensor->data());
    std::cout << "Sample values: ";
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < samples && i < tensor->shape().numel(); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << "..." << std::endl;
}

/**
 * @brief Example 1: Basic 2D Convolution
 */
void example_basic_conv2d() {
    std::cout << "=== Example 1: Basic 2D Convolution ===" << std::endl;
    std::cout << "Conv2D: 3x3 kernel, stride=1, padding=0" << std::endl << std::endl;
    
    // Create Conv2D layer: 3x3 kernel
    operators::Conv2DParam param(3, 3, 1, 1, 0, 0, 1, true);
    auto conv = std::make_shared<operators::Conv2D>(param);
    
    // Input: [1, 1, 5, 5]
    core::Shape input_shape({1, 1, 5, 5});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    
    float* input_data = static_cast<float*>(input->data());
    // Simple pattern: 1, 2, 3, ...
    for (int i = 0; i < 25; ++i) {
        input_data[i] = static_cast<float>(i + 1);
    }
    
    // Weight: [1, 1, 3, 3] (edge detection kernel)
    core::Shape weight_shape({1, 1, 3, 3});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);
    
    float* weight_data = static_cast<float*>(weight->data());
    // Sobel-like kernel
    weight_data[0] = -1.0f; weight_data[1] = 0.0f; weight_data[2] = 1.0f;
    weight_data[3] = -2.0f; weight_data[4] = 0.0f; weight_data[5] = 2.0f;
    weight_data[6] = -1.0f; weight_data[7] = 0.0f; weight_data[8] = 1.0f;
    
    // Bias: [1]
    core::Shape bias_shape({1});
    auto bias = core::Tensor::create(bias_shape, core::DataType::FLOAT32);
    float* bias_data = static_cast<float*>(bias->data());
    bias_data[0] = 0.0f;
    
    print_tensor_info(input, "Input");
    print_tensor_info(weight, "Weight");
    
    // Forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight, bias};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = conv->forward(inputs, outputs);
    
    if (status == core::Status::SUCCESS) {
        std::cout << "[SUCCESS] Forward pass successful!" << std::endl;
        print_tensor_info(outputs[0], "Output");
        print_output_sample(outputs[0]);
        std::cout << std::endl;
    } else {
        std::cout << "[ERROR] Forward pass failed!" << std::endl;
    }
}

/**
 * @brief Example 2: Convolution with Padding
 */
void example_conv_with_padding() {
    std::cout << "=== Example 2: Convolution with Padding ===" << std::endl;
    std::cout << "Conv2D: 3x3 kernel, stride=1, padding=1 (same)" << std::endl << std::endl;
    
    // Conv2D with padding=1 (SAME padding for 3x3 kernel)
    operators::Conv2DParam param(3, 3, 1, 1, 1, 1, 1, false);
    auto conv = std::make_shared<operators::Conv2D>(param);
    
    // Input: [1, 1, 4, 4]
    core::Shape input_shape({1, 1, 4, 4});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    
    float* input_data = static_cast<float*>(input->data());
    for (int i = 0; i < 16; ++i) {
        input_data[i] = 1.0f;
    }
    
    // Weight: [1, 1, 3, 3] (all ones - box filter)
    core::Shape weight_shape({1, 1, 3, 3});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);
    
    float* weight_data = static_cast<float*>(weight->data());
    for (int i = 0; i < 9; ++i) {
        weight_data[i] = 1.0f;
    }
    
    print_tensor_info(input, "Input");
    
    // Forward pass (no bias)
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = conv->forward(inputs, outputs);
    
    if (status == core::Status::SUCCESS) {
        std::cout << "[SUCCESS] Forward pass successful!" << std::endl;
        print_tensor_info(outputs[0], "Output");
        
        const float* output_data = static_cast<const float*>(outputs[0]->data());
        std::cout << "Corner pixel (sees 4 inputs): " << output_data[0] << std::endl;
        std::cout << "Center pixel (sees 9 inputs): " << output_data[5] << std::endl;
        std::cout << std::endl;
    }
}

/**
 * @brief Example 3: Multi-channel Convolution
 */
void example_multi_channel() {
    std::cout << "=== Example 3: Multi-channel Convolution ===" << std::endl;
    std::cout << "Input: 3 channels (RGB), Output: 16 channels" << std::endl << std::endl;
    
    // Conv2D: 3x3 kernel, 3->16 channels
    operators::Conv2DParam param(3, 3, 1, 1, 1, 1, 1, true);
    auto conv = std::make_shared<operators::Conv2D>(param);
    
    // Input: [1, 3, 8, 8] (simulating RGB image)
    core::Shape input_shape({1, 3, 8, 8});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    
    float* input_data = static_cast<float*>(input->data());
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < 192; ++i) {
        input_data[i] = dist(gen);
    }
    
    // Weight: [16, 3, 3, 3]
    core::Shape weight_shape({16, 3, 3, 3});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);
    
    float* weight_data = static_cast<float*>(weight->data());
    for (int i = 0; i < 432; ++i) {
        weight_data[i] = dist(gen) * 0.1f;
    }
    
    // Bias: [16]
    core::Shape bias_shape({16});
    auto bias = core::Tensor::create(bias_shape, core::DataType::FLOAT32);
    
    float* bias_data = static_cast<float*>(bias->data());
    for (int i = 0; i < 16; ++i) {
        bias_data[i] = dist(gen) * 0.01f;
    }
    
    print_tensor_info(input, "Input");
    print_tensor_info(weight, "Weight");
    
    // Forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight, bias};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = conv->forward(inputs, outputs);
    
    if (status == core::Status::SUCCESS) {
        std::cout << "[SUCCESS] Forward pass successful!" << std::endl;
        print_tensor_info(outputs[0], "Output");
        print_output_sample(outputs[0], 10);
        std::cout << std::endl;
    }
}

/**
 * @brief Example 4: Strided Convolution (Downsampling)
 */
void example_strided_conv() {
    std::cout << "=== Example 4: Strided Convolution ===" << std::endl;
    std::cout << "Conv2D: 3x3 kernel, stride=2 (downsampling)" << std::endl << std::endl;
    
    // Conv2D with stride=2
    operators::Conv2DParam param(3, 3, 2, 2, 1, 1, 1, false);
    auto conv = std::make_shared<operators::Conv2D>(param);
    
    // Input: [1, 1, 8, 8]
    core::Shape input_shape({1, 1, 8, 8});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    
    float* input_data = static_cast<float*>(input->data());
    for (int i = 0; i < 64; ++i) {
        input_data[i] = static_cast<float>(i % 10);
    }
    
    // Weight: [1, 1, 3, 3]
    core::Shape weight_shape({1, 1, 3, 3});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);
    
    float* weight_data = static_cast<float*>(weight->data());
    for (int i = 0; i < 9; ++i) {
        weight_data[i] = 0.11f;  // Averaging kernel
    }
    
    print_tensor_info(input, "Input");
    
    // Forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = conv->forward(inputs, outputs);
    
    if (status == core::Status::SUCCESS) {
        std::cout << "[SUCCESS] Forward pass successful!" << std::endl;
        print_tensor_info(outputs[0], "Output");
        std::cout << "Note: Output size is halved due to stride=2" << std::endl;
        print_output_sample(outputs[0]);
        std::cout << std::endl;
    }
}

/**
 * @brief Example 5: Convolution Network (Conv + ReLU)
 */
void example_conv_network() {
    std::cout << "=== Example 5: Convolution Network (Conv + ReLU) ===" << std::endl << std::endl;
    
    // Create layers
    operators::Conv2DParam param(3, 3, 1, 1, 1, 1, 1, true);
    auto conv = std::make_shared<operators::Conv2D>(param);
    auto relu = std::make_shared<operators::ReLU>();
    
    // Input: [1, 1, 4, 4]
    core::Shape input_shape({1, 1, 4, 4});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    
    float* input_data = static_cast<float*>(input->data());
    for (int i = 0; i < 16; ++i) {
        input_data[i] = static_cast<float>((i % 8) - 4);  // Mix of positive and negative
    }
    
    // Weight: [2, 1, 3, 3] (2 output channels)
    core::Shape weight_shape({2, 1, 3, 3});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);
    
    float* weight_data = static_cast<float*>(weight->data());
    for (int i = 0; i < 18; ++i) {
        weight_data[i] = (i < 9) ? 0.1f : -0.1f;
    }
    
    // Bias: [2]
    core::Shape bias_shape({2});
    auto bias = core::Tensor::create(bias_shape, core::DataType::FLOAT32);
    float* bias_data = static_cast<float*>(bias->data());
    bias_data[0] = 0.5f;
    bias_data[1] = -0.5f;
    
    print_tensor_info(input, "Input");
    
    // Forward through Conv
    std::vector<std::shared_ptr<core::Tensor>> conv_inputs = {input, weight, bias};
    std::vector<std::shared_ptr<core::Tensor>> conv_outputs;
    
    auto status = conv->forward(conv_inputs, conv_outputs);
    if (status != core::Status::SUCCESS) {
        std::cout << "[ERROR] Conv forward failed!" << std::endl;
        return;
    }
    
    print_tensor_info(conv_outputs[0], "After Conv");
    
    // Forward through ReLU
    std::vector<std::shared_ptr<core::Tensor>> relu_outputs;
    status = relu->forward(conv_outputs, relu_outputs);
    if (status != core::Status::SUCCESS) {
        std::cout << "[ERROR] ReLU forward failed!" << std::endl;
        return;
    }
    
    print_tensor_info(relu_outputs[0], "After ReLU");
    std::cout << "[SUCCESS] Conv + ReLU forward pass completed!" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Mini-Infer Conv2D Examples" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    try {
        example_basic_conv2d();
        std::cout << std::string(50, '-') << std::endl << std::endl;
        
        example_conv_with_padding();
        std::cout << std::string(50, '-') << std::endl << std::endl;
        
        example_multi_channel();
        std::cout << std::string(50, '-') << std::endl << std::endl;
        
        example_strided_conv();
        std::cout << std::string(50, '-') << std::endl << std::endl;
        
        example_conv_network();
        
        std::cout << "[SUCCESS] All examples completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}
