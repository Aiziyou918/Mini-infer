#include "mini_infer/core/tensor.h"
#include "mini_infer/utils/logger.h"
#include <iostream>
#include <cassert>

using namespace mini_infer;

void test_tensor_creation() {
    std::cout << "Testing tensor creation..." << std::endl;
    
    core::Shape shape({2, 3, 224, 224});
    auto tensor = core::Tensor::create(shape, core::DataType::FLOAT32);
    
    assert(tensor != nullptr);
    assert(tensor->shape().ndim() == 4);
    assert(tensor->shape()[0] == 2);
    assert(tensor->shape()[1] == 3);
    assert(!tensor->empty());
    
    std::cout << "✓ Tensor creation test passed" << std::endl;
}

void test_tensor_reshape() {
    std::cout << "Testing tensor reshape..." << std::endl;
    
    core::Shape shape({2, 3, 4});
    auto tensor = core::Tensor::create(shape, core::DataType::FLOAT32);
    
    core::Shape new_shape({6, 4});
    tensor->reshape(new_shape);
    
    assert(tensor->shape().ndim() == 2);
    assert(tensor->shape()[0] == 6);
    assert(tensor->shape()[1] == 4);
    
    std::cout << "✓ Tensor reshape test passed" << std::endl;
}

void test_shape() {
    std::cout << "Testing shape..." << std::endl;
    
    core::Shape shape({2, 3, 4, 5});
    assert(shape.ndim() == 4);
    assert(shape.numel() == 120);
    assert(shape[0] == 2);
    assert(shape[3] == 5);
    
    std::cout << "Shape: " << shape.to_string() << std::endl;
    std::cout << "✓ Shape test passed" << std::endl;
}

int main() {
    std::cout << "\n=== Mini-Infer Tensor Tests ===" << std::endl;
    
    try {
        test_shape();
        test_tensor_creation();
        test_tensor_reshape();
        
        std::cout << "\n✓ All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed: " << e.what() << std::endl;
        return 1;
    }
}

