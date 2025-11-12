#include "mini_infer/backends/cpu_backend.h"
#include "mini_infer/backends/backend.h"
#include <iostream>
#include <cassert>
#include <cstring>

using namespace mini_infer;

void test_cpu_backend_allocation() {
    std::cout << "Testing CPU backend allocation..." << std::endl;
    
    auto backend = backends::BackendFactory::create_backend(core::DeviceType::CPU);
    assert(backend != nullptr);
    assert(backend->device_type() == core::DeviceType::CPU);
    
    size_t size = 1024;
    void* ptr = backend->allocate(size);
    assert(ptr != nullptr);
    
    backend->memset(ptr, 0, size);
    backend->deallocate(ptr);
    
    std::cout << "✓ CPU backend allocation test passed" << std::endl;
}

void test_cpu_backend_memcpy() {
    std::cout << "Testing CPU backend memcpy..." << std::endl;
    
    auto backend = backends::BackendFactory::create_backend(core::DeviceType::CPU);
    
    size_t size = 100 * sizeof(float);
    void* src = backend->allocate(size);
    void* dst = backend->allocate(size);
    
    // 填充源数据
    float* src_data = static_cast<float*>(src);
    for (int i = 0; i < 100; ++i) {
        src_data[i] = static_cast<float>(i);
    }
    
    // 复制
    backend->memcpy(dst, src, size);
    
    // 验证
    float* dst_data = static_cast<float*>(dst);
    for (int i = 0; i < 100; ++i) {
        assert(dst_data[i] == static_cast<float>(i));
    }
    
    backend->deallocate(src);
    backend->deallocate(dst);
    
    std::cout << "✓ CPU backend memcpy test passed" << std::endl;
}

int main() {
    std::cout << "\n=== Mini-Infer Backend Tests ===" << std::endl;
    
    try {
        test_cpu_backend_allocation();
        test_cpu_backend_memcpy();
        
        std::cout << "\n✓ All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed: " << e.what() << std::endl;
        return 1;
    }
}

