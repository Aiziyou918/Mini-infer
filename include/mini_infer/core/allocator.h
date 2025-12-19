#pragma once

#include <cstddef>
#include <memory>

#if defined(MINI_INFER_DEBUG)
#include <mutex>
#include <unordered_map>
#endif

namespace mini_infer {
namespace core {

constexpr size_t kDefaultAlignment = 64;

/**
 * @brief Base class for memory allocators
 */
class Allocator {
public:
    virtual ~Allocator() = default;
    
    /**
     * @brief Allocate memory
     * @param size The size of the memory to allocate
     * @param alignment Desired memory alignment in bytes
     * @return A pointer to the allocated memory
     */
    virtual void* allocate(size_t size, size_t alignment = kDefaultAlignment) = 0;

    /**
     * @brief Deallocate memory
     * @note The pointer must have been allocated by the allocate method
     * @param ptr The pointer to the memory to deallocate
     */
    virtual void deallocate(void* ptr) = 0;
    
    /**
     * @brief Get the total amount of memory allocated (debug only)
     * @return The total amount of memory allocated
     */
    virtual size_t total_allocated() const { return 0; }
};

/**
 * @brief CPU memory allocator
 */
class CPUAllocator : public Allocator {
public:
    /**
     * @brief Allocate memory on the CPU
     * @param size The size of the memory to allocate
     * @return A pointer to the allocated memory
     */
    void* allocate(size_t size, size_t alignment = kDefaultAlignment) override;

    /**
     * @brief Deallocate memory on the CPU
     * @param ptr The pointer to the memory to deallocate
     */
    void deallocate(void* ptr) override;

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
     * @brief Get the instance of the CPUAllocator
     * @return The instance of the CPUAllocator
     */
    static CPUAllocator* get_instance();
    
private:
    CPUAllocator() = default;

#if defined(MINI_INFER_DEBUG)
    mutable std::mutex mutex_;                       ///< Mutex for thread safety
    std::unordered_map<void*, size_t> allocations_;  ///< Track allocation sizes
    size_t total_allocated_{0};                      ///< Current total allocated
    size_t peak_allocated_{0};                       ///< Peak memory usage
#endif
};

/**
 * @brief 分配器工厂
 */
class AllocatorFactory {
public:
    enum class AllocatorType {
        CPU,
        CUDA  // 未来支持
    };
    
    static Allocator* get_allocator(AllocatorType type);
};

} // namespace core
} // namespace mini_infer
