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

std::shared_ptr<Tensor> Tensor::create(const Shape& shape, DataType dtype) {
    return std::make_shared<Tensor>(shape, dtype);
}

} // namespace core
} // namespace mini_infer

