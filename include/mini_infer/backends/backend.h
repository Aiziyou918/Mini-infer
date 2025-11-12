#pragma once

#include "mini_infer/core/tensor.h"
#include "mini_infer/core/types.h"
#include <memory>

namespace mini_infer {
namespace backends {

/**
 * @brief Backend interface abstract class
 * Defines the interface that different compute backends (CPU, CUDA, etc.) need to implement
 */
class Backend {
public:
    virtual ~Backend() = default;
    
    /**
     * @brief Get the device type of the backend
     * @return The device type of the backend
     */
    virtual core::DeviceType device_type() const = 0;

    /**
     * @brief Allocate memory
     * @param size The size of the memory to allocate
     * @return A pointer to the allocated memory
     */
    virtual void* allocate(size_t size) = 0;

    /**
     * @brief Deallocate memory
     * @note The pointer must have been allocated by the allocate method
     */
    virtual void deallocate(void* ptr) = 0;

    /**
     * @brief Copy memory from one pointer to another
     * @param dst The destination pointer
     * @param src The source pointer
     * @param size The size of the memory to copy
     */
    virtual void memcpy(void* dst, const void* src, size_t size) = 0;

    /**
     * @brief Set memory to a value
     * @param ptr The pointer to the memory to set
     * @param value The value to set
     * @param size The size of the memory to set
     */
    virtual void memset(void* ptr, int value, size_t size) = 0;
    
    /**
     * @brief Copy a tensor from one pointer to another
     * @param dst The destination tensor
     * @param src The source tensor
     */
    virtual void copy_tensor(core::Tensor& dst, const core::Tensor& src) = 0;
    
    /**
     * @brief Synchronize the backend
     */
    virtual void synchronize() = 0;
    
    /**
     * @brief Get the name of the backend
     * @return The name of the backend
     */
    virtual const char* name() const = 0;
};

/**
 * @brief Backend factory
 */
class BackendFactory {
public:
    /**
     * @brief Create a backend
     * @param type The type of the backend to create
     * @return A shared pointer to the created backend
     */
    static std::shared_ptr<Backend> create_backend(core::DeviceType type);

    /**
     * @brief Get the default backend
     * @return A shared pointer to the default backend
     */
    static std::shared_ptr<Backend> get_default_backend();
};

} // namespace backends
} // namespace mini_infer

