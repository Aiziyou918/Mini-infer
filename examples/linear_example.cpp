#include "mini_infer/mini_infer.h"
#include <iostream>
#include <iomanip>
#include <random>

using namespace mini_infer;

/**
 * @brief Print tensor values in a formatted way
 */
void print_tensor(const std::shared_ptr<core::Tensor>& tensor, const std::string& name) {
    std::cout << name << " (shape: ";
    for (size_t i = 0; i < tensor->shape().ndim(); ++i) {
        std::cout << tensor->shape()[i];
        if (i < tensor->shape().ndim() - 1) std::cout << " x ";
    }
    std::cout << "):" << std::endl;
    
    if (tensor->dtype() == core::DataType::FLOAT32) {
        const float* data = static_cast<const float*>(tensor->data());
        size_t total = tensor->shape().numel();
        
        std::cout << std::fixed << std::setprecision(4);
        for (size_t i = 0; i < total; ++i) {
            std::cout << std::setw(10) << data[i];
            
            // Print newline based on last dimension
            if ((i + 1) % tensor->shape()[tensor->shape().ndim() - 1] == 0) {
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

/**
 * @brief Example 1: Basic Linear Layer
 */
void example_basic_linear() {
    std::cout << "=== Example 1: Basic Linear Layer ===" << std::endl;
    std::cout << "Linear: 4 features -> 3 features with bias" << std::endl << std::endl;
    
    // Create Linear layer: 4 input features -> 3 output features
    operators::LinearParam param(4, 3, true);
    auto linear = std::make_shared<operators::Linear>(param);
    
    // Create input tensor: [2, 4] (batch_size=2, in_features=4)
    core::Shape input_shape({2, 4});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    
    float* input_data = static_cast<float*>(input->data());
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f; input_data[3] = 4.0f;
    input_data[4] = 5.0f; input_data[5] = 6.0f; input_data[6] = 7.0f; input_data[7] = 8.0f;
    
    // Create weight tensor: [3, 4] (out_features=3, in_features=4)
    core::Shape weight_shape({3, 4});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);
    
    float* weight_data = static_cast<float*>(weight->data());
    // Initialize weights with simple pattern
    for (int i = 0; i < 12; ++i) {
        weight_data[i] = (i % 3 == 0) ? 1.0f : (i % 3 == 1) ? 0.5f : -0.5f;
    }
    
    // Create bias tensor: [3]
    core::Shape bias_shape({3});
    auto bias = core::Tensor::create(bias_shape, core::DataType::FLOAT32);
    
    float* bias_data = static_cast<float*>(bias->data());
    bias_data[0] = 0.1f;
    bias_data[1] = 0.2f;
    bias_data[2] = 0.3f;
    
    // Print inputs
    print_tensor(input, "Input");
    print_tensor(weight, "Weight");
    print_tensor(bias, "Bias");
    
    // Forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight, bias};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = linear->forward(inputs, outputs);
    
    if (status == core::Status::SUCCESS) {
        std::cout << "[SUCCESS] Forward pass successful!" << std::endl << std::endl;
        print_tensor(outputs[0], "Output");
    } else {
        std::cout << "[ERROR] Forward pass failed!" << std::endl;
    }
}

/**
 * @brief Example 2: Linear Layer without Bias
 */
void example_linear_without_bias() {
    std::cout << "=== Example 2: Linear Layer without Bias ===" << std::endl;
    std::cout << "Linear: 3 features -> 2 features (no bias)" << std::endl << std::endl;
    
    // Create Linear layer without bias
    operators::LinearParam param(3, 2, false);
    auto linear = std::make_shared<operators::Linear>(param);
    
    // Input: [1, 3]
    core::Shape input_shape({1, 3});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    
    float* input_data = static_cast<float*>(input->data());
    input_data[0] = 2.0f;
    input_data[1] = 3.0f;
    input_data[2] = 4.0f;
    
    // Weight: [2, 3]
    core::Shape weight_shape({2, 3});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);
    
    float* weight_data = static_cast<float*>(weight->data());
    weight_data[0] = 1.0f; weight_data[1] = 0.0f; weight_data[2] = -1.0f;
    weight_data[3] = 0.0f; weight_data[4] = 2.0f; weight_data[5] = 0.0f;
    
    print_tensor(input, "Input");
    print_tensor(weight, "Weight");
    
    // Forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = linear->forward(inputs, outputs);
    
    if (status == core::Status::SUCCESS) {
        std::cout << "[SUCCESS] Forward pass successful!" << std::endl << std::endl;
        print_tensor(outputs[0], "Output");
        
        // Manual calculation for verification
        // Output[0] = 2*1 + 3*0 + 4*(-1) = 2 - 4 = -2
        // Output[1] = 2*0 + 3*2 + 4*0 = 6
        std::cout << "Expected: [-2.0000, 6.0000]" << std::endl << std::endl;
    }
}

/**
 * @brief Example 3: Multi-Layer Perceptron (MLP)
 */
void example_mlp() {
    std::cout << "=== Example 3: Two-Layer MLP ===" << std::endl;
    std::cout << "Architecture: 4 -> 8 -> 2" << std::endl << std::endl;
    
    // Layer 1: 4 -> 8
    operators::LinearParam param1(4, 8, true);
    auto linear1 = std::make_shared<operators::Linear>(param1);
    
    // Layer 2: 8 -> 2
    operators::LinearParam param2(8, 2, true);
    auto linear2 = std::make_shared<operators::Linear>(param2);
    
    // ReLU activation
    auto relu = std::make_shared<operators::ReLU>();
    
    // Input: [1, 4]
    core::Shape input_shape({1, 4});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    
    float* input_data = static_cast<float*>(input->data());
    input_data[0] = 0.5f;
    input_data[1] = 1.0f;
    input_data[2] = 1.5f;
    input_data[3] = 2.0f;
    
    // Initialize weights and biases with random values
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    
    // Layer 1 weights: [8, 4]
    core::Shape weight1_shape({8, 4});
    auto weight1 = core::Tensor::create(weight1_shape, core::DataType::FLOAT32);
    float* weight1_data = static_cast<float*>(weight1->data());
    for (int i = 0; i < 32; ++i) {
        weight1_data[i] = dist(gen);
    }
    
    // Layer 1 bias: [8]
    core::Shape bias1_shape({8});
    auto bias1 = core::Tensor::create(bias1_shape, core::DataType::FLOAT32);
    float* bias1_data = static_cast<float*>(bias1->data());
    for (int i = 0; i < 8; ++i) {
        bias1_data[i] = dist(gen);
    }
    
    // Layer 2 weights: [2, 8]
    core::Shape weight2_shape({2, 8});
    auto weight2 = core::Tensor::create(weight2_shape, core::DataType::FLOAT32);
    float* weight2_data = static_cast<float*>(weight2->data());
    for (int i = 0; i < 16; ++i) {
        weight2_data[i] = dist(gen);
    }
    
    // Layer 2 bias: [2]
    core::Shape bias2_shape({2});
    auto bias2 = core::Tensor::create(bias2_shape, core::DataType::FLOAT32);
    float* bias2_data = static_cast<float*>(bias2->data());
    for (int i = 0; i < 2; ++i) {
        bias2_data[i] = dist(gen);
    }
    
    print_tensor(input, "Input");
    
    // Forward pass through layer 1
    std::vector<std::shared_ptr<core::Tensor>> layer1_inputs = {input, weight1, bias1};
    std::vector<std::shared_ptr<core::Tensor>> layer1_outputs;
    
    auto status = linear1->forward(layer1_inputs, layer1_outputs);
    if (status != core::Status::SUCCESS) {
        std::cout << "[ERROR] Layer 1 forward failed!" << std::endl;
        return;
    }
    
    print_tensor(layer1_outputs[0], "Layer 1 Output (before ReLU)");
    
    // Apply ReLU
    std::vector<std::shared_ptr<core::Tensor>> relu_outputs;
    status = relu->forward(layer1_outputs, relu_outputs);
    if (status != core::Status::SUCCESS) {
        std::cout << "[ERROR] ReLU forward failed!" << std::endl;
        return;
    }
    
    print_tensor(relu_outputs[0], "After ReLU");
    
    // Forward pass through layer 2
    std::vector<std::shared_ptr<core::Tensor>> layer2_inputs = {relu_outputs[0], weight2, bias2};
    std::vector<std::shared_ptr<core::Tensor>> layer2_outputs;
    
    status = linear2->forward(layer2_inputs, layer2_outputs);
    if (status != core::Status::SUCCESS) {
        std::cout << "[ERROR] Layer 2 forward failed!" << std::endl;
        return;
    }
    
    print_tensor(layer2_outputs[0], "Final Output");
    
    std::cout << "[SUCCESS] Two-layer MLP forward pass completed!" << std::endl << std::endl;
}

/**
 * @brief Example 4: Using Operator Factory
 */
void example_operator_factory() {
    std::cout << "=== Example 4: Using Operator Factory ===" << std::endl << std::endl;
    
    // Create Linear operator through factory
    auto linear = operators::OperatorFactory::create_operator("Linear");
    
    if (linear && linear->name() == "Linear") {
        std::cout << "[SUCCESS] Successfully created Linear operator via factory" << std::endl;
        std::cout << "  Operator name: " << linear->name() << std::endl << std::endl;
    } else {
        std::cout << "[ERROR] Failed to create Linear operator" << std::endl;
    }
}

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Mini-Infer Linear Layer Examples" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    try {
        example_basic_linear();
        std::cout << std::string(50, '-') << std::endl << std::endl;
        
        example_linear_without_bias();
        std::cout << std::string(50, '-') << std::endl << std::endl;
        
        example_mlp();
        std::cout << std::string(50, '-') << std::endl << std::endl;
        
        example_operator_factory();
        
        std::cout << "[SUCCESS] All examples completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}
