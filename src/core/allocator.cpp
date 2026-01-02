#include "mini_infer/core/allocator.h"
#include "mini_infer/backends/cpu/cpu_allocator.h"

#ifdef MINI_INFER_USE_CUDA
#include "mini_infer/backends/cuda/cuda_allocator.h"
#endif

namespace mini_infer {
namespace core {

// ============================================================================
// AllocatorFactory implementation
// ============================================================================

Allocator* AllocatorFactory::get_allocator(AllocatorType type) {
    switch (type) {
        case AllocatorType::CPU:
            return backends::cpu::CPUAllocator::instance();

        case AllocatorType::CUDA:
#ifdef MINI_INFER_USE_CUDA
            {
                // Create a static CUDA allocator instance
                static backends::cuda::CUDAAllocator cuda_allocator(0);
                return &cuda_allocator;
            }
#else
            // CUDA support not compiled
            return nullptr;
#endif

        default:
            return nullptr;
    }
}

} // namespace core
} // namespace mini_infer
