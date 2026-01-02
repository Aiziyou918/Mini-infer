#pragma once

#include <cstddef>
#include <memory>
#include "mini_infer/core/types.h"

namespace mini_infer {
namespace core {

constexpr size_t kDefaultAlignment = 64;

/**
 * @brief Base class for memory allocators
 *
 * Provides a unified interface for memory allocation across different devices.
 * Concrete implementations are in backends/cpu and backends/cuda.
 */
class Allocator {
public:
    virtual ~Allocator() = default;

    /**
     * @brief Allocate memory
     * @param size The size of the memory to allocate in bytes
     * @param alignment Desired memory alignment in bytes
     * @return A pointer to the allocated memory, or nullptr on failure
     */
    virtual void* allocate(size_t size, size_t alignment = kDefaultAlignment) = 0;

    /**
     * @brief Deallocate memory
     * @note The pointer must have been allocated by the allocate method
     * @param ptr The pointer to the memory to deallocate
     */
    virtual void deallocate(void* ptr) = 0;

    /**
     * @brief Get device type for this allocator
     * @return Device type (CPU, CUDA, etc.)
     */
    virtual DeviceType device_type() const = 0;

    /**
     * @brief Get the total amount of memory allocated (debug only)
     * @return The total amount of memory allocated in bytes
     */
    virtual size_t total_allocated() const { return 0; }
};

/**
 * @brief Allocator Factory
 *
 * Factory pattern for creating allocators for different device types.
 * Returns singleton instances for each allocator type.
 */
class AllocatorFactory {
public:
    enum class AllocatorType {
        CPU,   ///< CPU memory allocator
        CUDA   ///< CUDA device memory allocator
    };

    /**
     * @brief Get allocator for specified type
     * @param type Allocator type (CPU or CUDA)
     * @return Pointer to allocator instance, or nullptr if not available
     */
    static Allocator* get_allocator(AllocatorType type);
};

} // namespace core
} // namespace mini_infer
