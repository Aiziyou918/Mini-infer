#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace mini_infer {
namespace core {

/**
 * @brief Base class for memory allocators
 */
class Allocator {
public:
    virtual ~Allocator() = default;
    
    /**
     * @brief Allocate memory
     * @param size The size of the memory to allocate
     * @return A pointer to the allocated memory
     */
    virtual void* allocate(size_t size) = 0;

    /**
     * @brief Deallocate memory
     * @note The pointer must have been allocated by the allocate method
     * @param ptr The pointer to the memory to deallocate
     */
    virtual void deallocate(void* ptr) = 0;
    
    /**
     * @brief Get the total amount of memory allocated
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
    void* allocate(size_t size) override;

    /**
     * @brief Deallocate memory on the CPU
     * @param ptr The pointer to the memory to deallocate
     */
    void deallocate(void* ptr) override;

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
    
    /**
     * @brief Get the instance of the CPUAllocator
     * @return The instance of the CPUAllocator
     */
    static CPUAllocator* get_instance();
    
private:
    CPUAllocator() = default;
    
    mutable std::mutex mutex_;                       ///< Mutex for thread safety
    std::unordered_map<void*, size_t> allocations_;  ///< Track allocation sizes
    size_t total_allocated_{0};                      ///< Current total allocated
    size_t peak_allocated_{0};                       ///< Peak memory usage
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

