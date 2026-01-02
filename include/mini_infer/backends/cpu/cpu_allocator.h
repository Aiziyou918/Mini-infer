#pragma once

#include "mini_infer/core/allocator.h"

#if defined(MINI_INFER_DEBUG)
#include <mutex>
#include <unordered_map>
#endif

namespace mini_infer {
namespace backends {
namespace cpu {

/**
 * @brief CPU memory allocator
 *
 * Allocates aligned memory on the CPU using platform-specific APIs.
 * - Windows: _aligned_malloc/_aligned_free
 * - POSIX: posix_memalign/free
 *
 * In debug mode, tracks allocation statistics.
 */
class CPUAllocator : public core::Allocator {
public:
    /**
     * @brief Allocate aligned memory on the CPU
     * @param size The size of the memory to allocate in bytes
     * @param alignment Desired memory alignment in bytes (default: 64)
     * @return A pointer to the allocated memory, or nullptr on failure
     */
    void* allocate(size_t size, size_t alignment = core::kDefaultAlignment) override;

    /**
     * @brief Deallocate memory on the CPU
     * @param ptr The pointer to the memory to deallocate
     */
    void deallocate(void* ptr) override;

    /**
     * @brief Get device type
     * @return DeviceType::CPU
     */
    core::DeviceType device_type() const override {
        return core::DeviceType::CPU;
    }

#if defined(MINI_INFER_DEBUG)
    /**
     * @brief Get the total amount of memory currently allocated
     * @return The total amount of memory currently allocated in bytes
     */
    size_t total_allocated() const override;

    /**
     * @brief Get the peak memory usage
     * @return The peak memory usage in bytes
     */
    size_t peak_allocated() const { return peak_allocated_; }

    /**
     * @brief Get the number of active allocations
     * @return The number of active allocations
     */
    size_t allocation_count() const;
#else
    size_t total_allocated() const override { return 0; }
    size_t peak_allocated() const { return 0; }
    size_t allocation_count() const { return 0; }
#endif

    /**
     * @brief Get the singleton instance of CPUAllocator
     * @return Pointer to the singleton instance
     */
    static CPUAllocator* instance();

private:
    CPUAllocator() = default;
    ~CPUAllocator() = default;

    // Disable copy and move
    CPUAllocator(const CPUAllocator&) = delete;
    CPUAllocator& operator=(const CPUAllocator&) = delete;

#if defined(MINI_INFER_DEBUG)
    mutable std::mutex mutex_;                       ///< Mutex for thread safety
    std::unordered_map<void*, size_t> allocations_;  ///< Track allocation sizes
    size_t total_allocated_{0};                      ///< Current total allocated
    size_t peak_allocated_{0};                       ///< Peak memory usage
#endif
};

}  // namespace cpu
}  // namespace backends
}  // namespace mini_infer
