#pragma once

#include "mini_infer/core/allocator.h"
#include <cstddef>
#include <cstring>

namespace mini_infer {
namespace core {

/**
 * @brief RAII wrapper for allocator-managed memory
 * 
 * Provides a safe way to manage temporary buffers using the allocator system.
 * Similar to std::vector but uses CPUAllocator for unified memory management.
 * 
 * Usage:
 *   Buffer<float> buf(size);
 *   float* data = buf.data();
 */
template<typename T>
class Buffer {
public:
    /**
     * @brief Construct a buffer with the given size
     * @param size Number of elements (not bytes)
     * @param allocator The allocator to use (defaults to CPUAllocator)
     */
    explicit Buffer(size_t size, Allocator* allocator = nullptr)
        : size_(size)
        , allocator_(allocator ? allocator : CPUAllocator::get_instance())
        , data_(nullptr) {
        
        if (size_ > 0) {
            size_t bytes = size_ * sizeof(T);
            data_ = static_cast<T*>(allocator_->allocate(bytes));
            
            // Initialize to zero
            if (data_) {
                std::memset(data_, 0, bytes);
            }
        }
    }
    
    /**
     * @brief Destructor - deallocates the buffer
     */
    ~Buffer() {
        if (data_) {
            allocator_->deallocate(data_);
            data_ = nullptr;
        }
    }
    
    // Disable copy
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    
    // Enable move
    Buffer(Buffer&& other) noexcept
        : size_(other.size_)
        , allocator_(other.allocator_)
        , data_(other.data_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            // Deallocate current data
            if (data_) {
                allocator_->deallocate(data_);
            }
            
            // Move from other
            size_ = other.size_;
            allocator_ = other.allocator_;
            data_ = other.data_;
            
            // Clear other
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    /**
     * @brief Get the data pointer
     * @return Pointer to the data
     */
    T* data() { return data_; }
    const T* data() const { return data_; }
    
    /**
     * @brief Get the size in elements
     * @return Number of elements
     */
    size_t size() const { return size_; }
    
    /**
     * @brief Get the size in bytes
     * @return Size in bytes
     */
    size_t size_in_bytes() const { return size_ * sizeof(T); }
    
    /**
     * @brief Check if the buffer is empty
     * @return True if empty
     */
    bool empty() const { return data_ == nullptr || size_ == 0; }
    
    /**
     * @brief Element access operator
     * @param index Element index
     * @return Reference to element
     */
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }
    
private:
    size_t size_;           ///< Number of elements
    Allocator* allocator_;  ///< The allocator used
    T* data_;               ///< Pointer to the data
};

} // namespace core
} // namespace mini_infer
