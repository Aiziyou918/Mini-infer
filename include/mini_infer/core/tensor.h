#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mini_infer/core/types.h"
#include "mini_infer/core/allocator.h"

namespace mini_infer {
namespace core {

/**
 * @brief Data Type Enumeration
 */
enum class DataType {
    FLOAT32,  ///< 32-bit floating point
    FLOAT16,  ///< 16-bit floating point
    INT32,    ///< 32-bit integer
    INT64,    ///< 64-bit integer
    INT8,     ///< 8-bit integer
    UINT8,    ///< 8-bit unsigned integer
    BOOL,     ///< Boolean
};

/**
 * @brief Tensor Shape Class
 */
class Shape {
   public:
    Shape() = default;
    explicit Shape(const std::vector<int64_t>& dims);

    /**
     * @brief Get the dimension at the given index
     *
     * @param index The index of the dimension
     * @return The dimension at the given index
     */
    int64_t operator[](size_t index) const;

    /**
     * @brief Get the number of dimensions of the tensor
     * @return The number of dimensions of the tensor
     */
    size_t ndim() const {
        return dims_.size();
    }

    /**
     * @brief Get the number of elements in the tensor
     *
     * @return The number of elements in the tensor
     */
    int64_t numel() const;

    /**
     * @brief Get the dimensions of the tensor
     * @return The dimensions of the tensor
     */
    const std::vector<int64_t>& dims() const {
        return dims_;
    }

    /**
     * @brief Convert the tensor shape to a string
     *
     * vector<int64_t> {dim1, dim2, dim3} --> string "[dim1, dim2, dim3]"
     *
     * @return The string representation of the tensor shape
     */
    std::string to_string() const;

    /**
     * @brief Check if shape has dynamic dimensions (dimensions with value -1)
     *
     * @return True if any dimension is -1 (dynamic), false otherwise
     */
    bool is_dynamic() const;

    /**
     * @brief Equality comparison
     */
    bool operator==(const Shape& other) const;
    bool operator!=(const Shape& other) const {
        return !(*this == other);
    }

   private:
    std::vector<int64_t> dims_;  ///< The dimensions of the tensor
};

/**
 * @brief Tensor Class - The basic data structure of the inference framework
 */
class Storage {
   public:
    Storage() = default;
    Storage(size_t capacity_bytes, DeviceType device, size_t alignment = kDefaultAlignment);
    Storage(const std::shared_ptr<void>& external, size_t capacity_bytes,
            DeviceType device = DeviceType::CPU);
    ~Storage() = default;

    void reset(size_t capacity_bytes, DeviceType device, size_t alignment = kDefaultAlignment);
    // The caller must provide a shared_ptr with the correct deleter for the external buffer.
    void set_external(const std::shared_ptr<void>& external, size_t capacity_bytes,
                      DeviceType device);

    void* data() const {
        return buffer_.get();
    }

    size_t capacity() const {
        return capacity_;
    }

    DeviceType device() const {
        return device_;
    }

    bool empty() const {
        return buffer_ == nullptr;
    }

   private:
    std::shared_ptr<void> buffer_;
    size_t capacity_{0};
    DeviceType device_{DeviceType::CPU};
};

class Tensor {
   public:
    Tensor() = default;
    Tensor(const Shape& shape, DataType dtype, DeviceType device = DeviceType::CPU,
           size_t alignment = kDefaultAlignment);
    ~Tensor() = default;

    // Disable copy, allow move
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    const Shape& shape() const {
        return shape_;
    }

    DataType dtype() const {
        return dtype_;
    }

    DeviceType device() const {
        return device_;
    }

    const std::vector<int64_t>& strides() const {
        return strides_;
    }

    size_t storage_offset() const {
        return storage_offset_;
    }

    void* data();
    const void* data() const;

    size_t size_in_bytes() const;

    bool empty() const {
        return !storage_ || storage_->empty();
    }

    void reshape(const Shape& new_shape);
    void resize(const Shape& new_shape);

    size_t capacity() const {
        return storage_ ? storage_->capacity() : 0;
    }

    // Reshape-only view: shares storage and keeps the same offset/contiguous layout.
    std::shared_ptr<Tensor> view(const Shape& new_shape) const;

    static std::shared_ptr<Tensor> create(const Shape& shape, DataType dtype,
                                          DeviceType device = DeviceType::CPU);

    /**
     * @brief Bind an externally allocated buffer to this tensor.
     *
     * @param data Shared ownership of the external buffer
     * @param capacity_bytes Capacity of the external buffer in bytes
     * @param device Device where the external buffer resides
     *
     * Used by the runtime memory planner to let multiple tensors share
     * the same preallocated memory pool.
     */
    void bind_external_data(const std::shared_ptr<void>& data, size_t capacity_bytes,
                            DeviceType device = DeviceType::CPU);
    bool bind_external_data_with_offset(const std::shared_ptr<void>& data, size_t capacity_bytes,
                                        size_t offset_bytes,
                                        DeviceType device = DeviceType::CPU);

    void set_shape_metadata(const Shape& shape);

    void set_dtype(DataType dtype) {
        dtype_ = dtype;
    }

    size_t element_size() const;

   private:
    void allocate();
    void ensure_contiguous_storage(size_t new_size_bytes);
    void compute_contiguous_strides();

    Shape shape_;
    DataType dtype_{DataType::FLOAT32};
    std::shared_ptr<Storage> storage_;
    size_t storage_offset_{0};
    std::vector<int64_t> strides_;
    DeviceType device_{DeviceType::CPU};
    size_t alignment_{kDefaultAlignment};
};

}  // namespace core
}  // namespace mini_infer
