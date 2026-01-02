#include "mini_infer/backends/cuda/cuda_allocator.h"

#ifdef MINI_INFER_USE_CUDA

#include "mini_infer/utils/logger.h"
#include <cuda_runtime.h>

namespace mini_infer {
namespace backends {
namespace cuda {

CUDAAllocator::CUDAAllocator(int device_id) : device_id_(device_id) {
    cudaError_t status = cudaSetDevice(device_id_);
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDAAllocator] Failed to set device " +
                     std::to_string(device_id_) + ": " +
                     std::string(cudaGetErrorString(status)));
        throw std::runtime_error("Failed to set CUDA device");
    }

    MI_LOG_INFO("[CUDAAllocator] Initialized for device " + std::to_string(device_id_));
}

void* CUDAAllocator::allocate(size_t size, size_t alignment) {
    // Note: alignment parameter is currently ignored
    // CUDA uses 256-byte alignment by default which is sufficient for most cases
    (void)alignment;

    if (size == 0) {
        MI_LOG_WARNING("[CUDAAllocator] Attempted to allocate 0 bytes");
        return nullptr;
    }

    void* ptr = nullptr;
    cudaError_t status = cudaMalloc(&ptr, size);

    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDAAllocator] Failed to allocate " +
                     std::to_string(size) + " bytes: " +
                     std::string(cudaGetErrorString(status)));
        return nullptr;
    }

    MI_LOG_DEBUG("[CUDAAllocator] Allocated " +
                 std::to_string(size / 1024.0 / 1024.0) + " MB at " +
                 std::to_string(reinterpret_cast<uintptr_t>(ptr)));

    return ptr;
}

void CUDAAllocator::deallocate(void* ptr) {
    if (!ptr) {
        return;
    }

    cudaError_t status = cudaFree(ptr);
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDAAllocator] Failed to deallocate memory at " +
                     std::to_string(reinterpret_cast<uintptr_t>(ptr)) + ": " +
                     std::string(cudaGetErrorString(status)));
    } else {
        MI_LOG_DEBUG("[CUDAAllocator] Deallocated memory at " +
                     std::to_string(reinterpret_cast<uintptr_t>(ptr)));
    }
}

}  // namespace cuda
}  // namespace backends
}  // namespace mini_infer

#endif  // MINI_INFER_USE_CUDA
