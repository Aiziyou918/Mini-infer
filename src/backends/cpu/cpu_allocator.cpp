#include "mini_infer/backends/cpu/cpu_allocator.h"

#include <cstdlib>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace mini_infer {
namespace backends {
namespace cpu {

void* CPUAllocator::allocate(size_t size, size_t alignment) {
    if (size == 0) {
        return nullptr;
    }

    if (alignment == 0) {
        alignment = core::kDefaultAlignment;
    }

#if defined(_MSC_VER)
    void* ptr = _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    if (posix_memalign(&ptr, alignment, aligned_size) != 0) {
        ptr = nullptr;
    }
#endif

#if defined(MINI_INFER_DEBUG)
    if (ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_[ptr] = size;
        total_allocated_ += size;
        if (total_allocated_ > peak_allocated_) {
            peak_allocated_ = total_allocated_;
        }
    }
#endif

    return ptr;
}

void CPUAllocator::deallocate(void* ptr) {
    if (ptr) {
#if defined(MINI_INFER_DEBUG)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = allocations_.find(ptr);
            if (it != allocations_.end()) {
                total_allocated_ -= it->second;
                allocations_.erase(it);
            }
        }
#endif

#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
}

#if defined(MINI_INFER_DEBUG)
size_t CPUAllocator::total_allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_;
}

size_t CPUAllocator::allocation_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocations_.size();
}
#endif

CPUAllocator* CPUAllocator::instance() {
    static CPUAllocator instance;
    return &instance;
}

}  // namespace cpu
}  // namespace backends
}  // namespace mini_infer
