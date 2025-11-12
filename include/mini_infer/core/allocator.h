#pragma once

#include <cstddef>
#include <memory>

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
     * @brief Get the total amount of memory allocated on the CPU
     * @return The total amount of memory allocated on the CPU
     */
    size_t total_allocated() const override { return total_allocated_; }
    
    /**
     * @brief Get the instance of the CPUAllocator
     * @return The instance of the CPUAllocator
     */
    static CPUAllocator* get_instance();
    
private:
    CPUAllocator() = default;
    size_t total_allocated_{0};
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

