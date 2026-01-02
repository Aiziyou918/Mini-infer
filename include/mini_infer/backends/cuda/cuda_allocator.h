#pragma once

#include "mini_infer/core/allocator.h"

#ifdef MINI_INFER_USE_CUDA

#include <cuda_runtime.h>

namespace mini_infer {
namespace backends {
namespace cuda {

/**
 * @brief CUDA Memory Allocator
 *
 * Simple CUDA memory allocator using cudaMalloc/cudaFree.
 * Phase 1 implementation - direct allocation without pooling.
 *
 * Future enhancements (Phase 3+):
 * - Memory pooling for reduced allocation overhead
 * - Caching allocator (similar to PyTorch's caching allocator)
 * - Memory statistics and tracking
 */
class CUDAAllocator : public core::Allocator {
public:
    /**
     * @brief Construct CUDA allocator
     *
     * @param device_id CUDA device ID (default: 0)
     */
    explicit CUDAAllocator(int device_id = 0);

    /**
     * @brief Destructor
     */
    ~CUDAAllocator() override = default;

    // Disable copy
    CUDAAllocator(const CUDAAllocator&) = delete;
    CUDAAllocator& operator=(const CUDAAllocator&) = delete;

    /**
     * @brief Allocate CUDA device memory
     *
     * @param size Size in bytes to allocate
     * @param alignment Memory alignment (currently ignored, CUDA uses 256-byte alignment)
     * @return Pointer to allocated memory, or nullptr on failure
     */
    void* allocate(size_t size, size_t alignment) override;

    /**
     * @brief Deallocate CUDA device memory
     *
     * @param ptr Pointer to memory to deallocate
     */
    void deallocate(void* ptr) override;

    /**
     * @brief Get device type
     * @return DeviceType::CUDA
     */
    core::DeviceType device_type() const override {
        return core::DeviceType::CUDA;
    }

    /**
     * @brief Get CUDA device ID
     * @return Device ID
     */
    int device_id() const { return device_id_; }

private:
    int device_id_;  ///< CUDA device ID
};

}  // namespace cuda
}  // namespace backends
}  // namespace mini_infer

#endif  // MINI_INFER_USE_CUDA
