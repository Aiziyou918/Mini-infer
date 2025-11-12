#pragma once

#include <memory>
#include <vector>
#include <string>
#include <cstdint>

namespace mini_infer {
namespace core {

/**
 * @brief Data Type Enumeration
 */
enum class DataType {
    FLOAT32, ///< 32-bit floating point
    FLOAT16, ///< 16-bit floating point
    INT32, ///< 32-bit integer
    INT8, ///< 8-bit integer
    UINT8, ///< 8-bit unsigned integer
    BOOL, ///< Boolean
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
    size_t ndim() const { return dims_.size(); }

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
    const std::vector<int64_t>& dims() const { return dims_; }

    /**
     * @brief Convert the tensor shape to a string
     * 
     * vector<int64_t> {dim1, dim2, dim3} --> string "[dim1, dim2, dim3]"
     * 
     * @return The string representation of the tensor shape
     */
    std::string to_string() const;
    
private:
    std::vector<int64_t> dims_; ///< The dimensions of the tensor
};

/**
 * @brief Tensor Class - The basic data structure of the inference framework
 */
class Tensor {
public:
    Tensor() = default;
    Tensor(const Shape& shape, DataType dtype);
    ~Tensor() = default;
    
    // Disable copy, allow move
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    
    // Accessors
    /**
     * @brief Get the shape of the tensor
     * @return The shape of the tensor
     */
    const Shape& shape() const { return shape_; }

    /**
     * @brief Get the data type of the tensor
     * @return The data type of the tensor
     */
    DataType dtype() const { return dtype_; }

    /**
     * @brief Get the data of the tensor
     * @return The data of the tensor
     */
    void* data() { return data_.get(); }

    /**
     * @brief Get the data of the tensor
     * @return The data of the tensor
     */
    const void* data() const { return data_.get(); }

    /**
     * @brief Get the size of the tensor in bytes
     * 
     * @return The size of the tensor in bytes
     */
    size_t size_in_bytes() const;
    
    // Utility methods
    /**
     * @brief Check if the tensor is empty
     * @return True if the tensor is empty, false otherwise
     */
    bool empty() const { return data_ == nullptr; }

    /**
     * @brief Reshape the tensor
     * @param new_shape The new shape of the tensor
     */
    void reshape(const Shape& new_shape);

    /**
     * @brief Create a tensor
     * @param shape The shape of the tensor
     * @param dtype The data type of the tensor
     * @return The created tensor
     */
    static std::shared_ptr<Tensor> create(const Shape& shape, DataType dtype);
    
private:
    Shape shape_; ///< The shape of the tensor
    DataType dtype_{DataType::FLOAT32}; ///< The data type of the tensor
    std::shared_ptr<void> data_; ///< The data of the tensor
    
    /**
     * @brief Allocate the memory for the tensor
     * @return The size of the tensor in bytes
     */
    void allocate();

    /**
     * @brief Get the size of the element in the tensor
     * @return The size of the element in the tensor
     */
    size_t element_size() const;
};

} // namespace core
} // namespace mini_infer

