#include "mini_infer/mini_infer.h"
#include <iostream>
#include <iomanip>

using namespace mini_infer;

int main() {
    std::cout << "=== ReLU Operator Example ===" << std::endl;
    std::cout << "Mini-Infer Version: " << get_version() << std::endl;
    std::cout << std::endl;
    
    // 1. 创建输入张量
    std::cout << "1. Creating input tensor..." << std::endl;
    core::Shape shape({1, 5});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    
    // 2. 填充数据
    std::cout << "2. Filling input data: [-2.0, -1.0, 0.0, 1.0, 2.0]" << std::endl;
    float* input_data = static_cast<float*>(input->data());
    input_data[0] = -2.0f;
    input_data[1] = -1.0f;
    input_data[2] = 0.0f;
    input_data[3] = 1.0f;
    input_data[4] = 2.0f;
    
    // 3. 创建 ReLU 算子（方式1：直接创建）
    std::cout << "3. Creating ReLU operator..." << std::endl;
    auto relu = std::make_shared<operators::ReLU>();
    
    // 4. 执行前向计算
    std::cout << "4. Running forward pass..." << std::endl;
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = relu->forward(inputs, outputs);
    if (status != core::Status::SUCCESS) {
        std::cerr << "Error: Forward pass failed!" << std::endl;
        return 1;
    }
    
    // 5. 获取输出
    std::cout << "5. Getting output..." << std::endl;
    auto output = outputs[0];
    float* output_data = static_cast<float*>(output->data());
    
    // 6. 打印结果
    std::cout << "\nResults:" << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "Input:  [";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << std::setw(6) << input_data[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << " ]" << std::endl;
    
    std::cout << "Output: [";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << std::setw(6) << output_data[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << " ]" << std::endl;
    
    std::cout << "\nExpected: [  0.00,   0.00,   0.00,   1.00,   2.00 ]" << std::endl;
    
    // 7. 使用工厂创建算子（方式2：通过工厂）
    std::cout << "\n--- Using Operator Factory ---" << std::endl;
    auto relu_from_factory = operators::OperatorFactory::create_operator("ReLU");
    std::cout << "Created operator: " << relu_from_factory->name() << std::endl;
    
    // 8. 测试形状推断
    std::cout << "\n--- Testing Shape Inference ---" << std::endl;
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    
    status = relu->infer_shape(input_shapes, output_shapes);
    if (status == core::Status::SUCCESS) {
        std::cout << "Input shape: [";
        for (size_t i = 0; i < input_shapes[0].ndim(); ++i) {
            std::cout << input_shapes[0][i];
            if (i < input_shapes[0].ndim() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "Output shape: [";
        for (size_t i = 0; i < output_shapes[0].ndim(); ++i) {
            std::cout << output_shapes[0][i];
            if (i < output_shapes[0].ndim() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "\n[SUCCESS] ReLU example completed successfully!" << std::endl;
    return 0;
}
