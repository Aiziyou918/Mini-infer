#include "mini_infer/core/tensor.h"
#include "mini_infer/core/buffer.h"
#include "mini_infer/core/allocator.h"
#include "mini_infer/backends/cpu/cpu_allocator.h"
#include "mini_infer/operators/conv2d.h"
#include <iostream>
#include <iomanip>

using namespace mini_infer;

/**
 * @brief Print memory statistics
 */
void print_memory_stats(const std::string& label) {
    auto allocator = backends::cpu::CPUAllocator::instance();
    
    double current_mb = allocator->total_allocated() / 1024.0 / 1024.0;
    double peak_mb = allocator->peak_allocated() / 1024.0 / 1024.0;
    size_t count = allocator->allocation_count();
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[" << label << "]" << std::endl;
    std::cout << "  Current: " << current_mb << " MB" << std::endl;
    std::cout << "  Peak:    " << peak_mb << " MB" << std::endl;
    std::cout << "  Active allocations: " << count << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Demonstrate memory tracking with Tensor
 */
void test_tensor_memory() {
    std::cout << "=== Tensor Memory Tracking ===" << std::endl;
    
    print_memory_stats("Initial");
    
    {
        // Create a large tensor
        auto tensor1 = core::Tensor::create({100, 100, 100}, core::DataType::FLOAT32);
        print_memory_stats("After creating tensor1 [100x100x100 float32]");
        
        // Create another tensor
        auto tensor2 = core::Tensor::create({200, 200, 50}, core::DataType::FLOAT32);
        print_memory_stats("After creating tensor2 [200x200x50 float32]");
        
    } // tensors destroyed here
    
    print_memory_stats("After tensors destroyed");
}

/**
 * @brief Demonstrate memory tracking with Buffer
 */
void test_buffer_memory() {
    std::cout << "=== Buffer Memory Tracking ===" << std::endl;
    
    print_memory_stats("Initial");
    
    {
        // Allocate buffer
        core::Buffer<float> buffer1(1024 * 1024);  // 4 MB
        print_memory_stats("After allocating buffer1 [4 MB]");
        
        {
            // Nested scope
            core::Buffer<int32_t> buffer2(512 * 1024);  // 2 MB
            print_memory_stats("After allocating buffer2 [2 MB]");
            
        } // buffer2 destroyed
        
        print_memory_stats("After buffer2 destroyed");
        
    } // buffer1 destroyed
    
    print_memory_stats("After buffer1 destroyed");
}

/**
 * @brief Demonstrate memory tracking with Conv2D
 */
void test_conv2d_memory() {
    std::cout << "=== Conv2D Memory Tracking ===" << std::endl;
    
    print_memory_stats("Initial");
    
    // Create Conv2D operator
    operators::Conv2DParam param;
    param.kernel_h = 3;
    param.kernel_w = 3;
    param.stride_h = 1;
    param.stride_w = 1;
    param.padding_h = 1;
    param.padding_w = 1;
    param.use_bias = true;
    
    auto conv = std::make_shared<operators::Conv2D>(param);
    
    // Create input tensors
    auto input = core::Tensor::create({8, 64, 56, 56}, core::DataType::FLOAT32);
    auto weight = core::Tensor::create({128, 64, 3, 3}, core::DataType::FLOAT32);
    auto bias = core::Tensor::create({128}, core::DataType::FLOAT32);
    
    print_memory_stats("After creating input/weight/bias tensors");
    
    // Forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight, bias};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    {
        auto status = conv->forward(inputs, outputs);
        if (status != core::Status::SUCCESS) {
            std::cerr << "Conv2D forward failed!" << std::endl;
            return;
        }
        
        print_memory_stats("During forward (col_buffer allocated)");
        
    } // col_buffer destroyed here
    
    print_memory_stats("After forward completed");
    
    std::cout << "Output shape: [";
    for (size_t i = 0; i < outputs[0]->shape().ndim(); ++i) {
        std::cout << outputs[0]->shape()[i];
        if (i < outputs[0]->shape().ndim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

/**
 * @brief Demonstrate memory leak detection
 */
void test_memory_leak_detection() {
    std::cout << "=== Memory Leak Detection ===" << std::endl;
    
    auto allocator = core::CPUAllocator::get_instance();
    
    size_t initial_count = allocator->allocation_count();
    size_t initial_size = allocator->total_allocated();
    
    print_memory_stats("Before test");
    
    {
        // Simulate some operations
        auto tensor = core::Tensor::create({1000, 1000}, core::DataType::FLOAT32);
        core::Buffer<float> buffer(1024 * 1024);
        
        print_memory_stats("During test");
        
    } // All memory should be freed
    
    print_memory_stats("After test");
    
    size_t final_count = allocator->allocation_count();
    size_t final_size = allocator->total_allocated();
    
    std::cout << "Leak check:" << std::endl;
    if (final_count == initial_count && final_size == initial_size) {
        std::cout << "  [SUCCESS] No memory leak detected!" << std::endl;
    } else {
        std::cout << "  [FAILED] Memory leak detected!" << std::endl;
        std::cout << "    Leaked allocations: " << (final_count - initial_count) << std::endl;
        std::cout << "    Leaked bytes: " << (final_size - initial_size) << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "Mini-Infer Memory Tracking Example" << std::endl;
    std::cout << "====================================" << std::endl << std::endl;
    
    // Test different scenarios
    test_tensor_memory();
    test_buffer_memory();
    test_conv2d_memory();
    test_memory_leak_detection();
    
    std::cout << "All tests completed!" << std::endl;
    
    return 0;
}
