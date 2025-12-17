#include "mini_infer/core/tensor.h"
#include "mini_infer/core/allocator.h"
#include <sstream>
#include <numeric>
#include <cstring>

namespace mini_infer {
namespace core {

// ============================================================================
// Shape implementation
// ============================================================================

Shape::Shape(const std::vector<int64_t>& dims) : dims_(dims) {}

int64_t Shape::operator[](size_t index) const {
    return dims_[index];
}

int64_t Shape::numel() const {
    if (dims_.empty()) return 0;
    return std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<int64_t>());
}

std::string Shape::to_string() const {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims_.size(); ++i) {
        oss << dims_[i];
        if (i < dims_.size() - 1) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

bool Shape::is_dynamic() const {
    for (int64_t dim : dims_) {
        if (dim < 0) {
            return true;
        }
    }
    return false;
}

bool Shape::operator==(const Shape& other) const {
    return dims_ == other.dims_;
}

// ============================================================================
// Tensor implementation
// ============================================================================

Tensor::Tensor(const Shape& shape, DataType dtype)
    : shape_(shape), dtype_(dtype) {
    allocate();
}

size_t Tensor::element_size() const {
    switch (dtype_) {
        case DataType::FLOAT32: return sizeof(float);
        case DataType::FLOAT16: return 2;
        case DataType::INT32: return sizeof(int32_t);
        case DataType::INT64: return sizeof(int64_t);
        case DataType::INT8: return sizeof(int8_t);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::BOOL: return sizeof(bool);
        default: return 0;
    }
}

size_t Tensor::size_in_bytes() const {
    return shape_.numel() * element_size();
}

void Tensor::allocate() {
    size_t bytes = size_in_bytes();
    if (bytes > 0) {
        void* ptr = CPUAllocator::get_instance()->allocate(bytes);
        data_ = std::shared_ptr<void>(ptr, [](void* p) {
            CPUAllocator::get_instance()->deallocate(p);
        });
        // Initialize to 0
        std::memset(ptr, 0, bytes);
    }
}

void Tensor::reshape(const Shape& new_shape) {
    if (new_shape.numel() != shape_.numel()) {
        // The number of elements must be the same
        return;
    }
    shape_ = new_shape;
}

std::shared_ptr<Tensor> Tensor::view(const Shape& new_shape) const {
    // Check that the total number of elements matches
    if (new_shape.numel() != shape_.numel()) {
        return nullptr;  // Invalid view
    }
    
    // Create a new Tensor object
    auto view_tensor = std::make_shared<Tensor>();
    view_tensor->shape_ = new_shape;
    view_tensor->dtype_ = dtype_;
    view_tensor->data_ = data_;  // Share the same data pointer (zero-copy!)
    
    return view_tensor;
}

std::shared_ptr<Tensor> Tensor::create(const Shape& shape, DataType dtype) {
    return std::make_shared<Tensor>(shape, dtype);
}

void Tensor::set_shape_metadata(const Shape& shape) {
    shape_ = shape;
}

} // namespace core
} // namespace mini_infer

