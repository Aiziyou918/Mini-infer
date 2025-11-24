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
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_[ptr] = size;
        total_allocated_ += size;
        if (total_allocated_ > peak_allocated_) {
            peak_allocated_ = total_allocated_;
        }
    }
    return ptr;
}

void CPUAllocator::deallocate(void* ptr) {
    if (ptr) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = allocations_.find(ptr);
            if (it != allocations_.end()) {
                total_allocated_ -= it->second;
                allocations_.erase(it);
            }
        }
        std::free(ptr);
    }
}

size_t CPUAllocator::total_allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_;
}

size_t CPUAllocator::allocation_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocations_.size();
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

