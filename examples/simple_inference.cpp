#include "mini_infer/core/tensor.h"
#include "mini_infer/backends/backend.h"
#include "mini_infer/utils/logger.h"
#include <iostream>

using namespace mini_infer;

int main() {
    std::cout << "\n=== Mini-Infer Simple Inference Example ===" << std::endl;
    
    // 设置日志级别
    utils::Logger::get_instance().set_level(utils::LogLevel::INFO);
    
    // 创建张量
    MI_LOG_INFO("Creating tensors...");
    core::Shape shape({1, 3, 224, 224});
    auto input_tensor = core::Tensor::create(shape, core::DataType::FLOAT32);
    
    std::cout << "Input shape: " << input_tensor->shape().to_string() << std::endl;
    std::cout << "Input size: " << input_tensor->size_in_bytes() << " bytes" << std::endl;
    
    // 创建后端
    MI_LOG_INFO("Creating CPU backend...");
    auto backend = backends::BackendFactory::get_default_backend();
    std::cout << "Backend: " << backend->name() << std::endl;
    
    // 模拟填充输入数据
    MI_LOG_INFO("Filling input data...");
    float* data = static_cast<float*>(input_tensor->data());
    for (int64_t i = 0; i < input_tensor->shape().numel(); ++i) {
        data[i] = 0.5f;
    }
    
    MI_LOG_INFO("Simple inference example completed successfully!");
    
    return 0;
}

