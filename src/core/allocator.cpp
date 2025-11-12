#include "mini_infer/core/allocator.h"
#include <cstdlib>
#include <cstring>

namespace mini_infer {
namespace core {

// ============================================================================
// CPUAllocator implementation
// ============================================================================

void* CPUAllocator::allocate(size_t size) {
    if (size == 0) return nullptr;
    
    void* ptr = std::malloc(size);
    if (ptr) {
        total_allocated_ += size;
    }
    return ptr;
}

void CPUAllocator::deallocate(void* ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

CPUAllocator* CPUAllocator::get_instance() {
    static CPUAllocator instance;
    return &instance;
}

// ============================================================================
// AllocatorFactory implementation
// ============================================================================

Allocator* AllocatorFactory::get_allocator(AllocatorType type) {
    switch (type) {
        case AllocatorType::CPU:
            return CPUAllocator::get_instance();
        case AllocatorType::CUDA:
            // TODO: Implement CUDA allocator
            return nullptr;
        default:
            return nullptr;
    }
}

} // namespace core
} // namespace mini_infer

