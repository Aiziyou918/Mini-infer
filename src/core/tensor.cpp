#include "mini_infer/core/tensor.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <sstream>

#include "mini_infer/core/allocator.h"
#include "mini_infer/utils/logger.h"

#ifdef MINI_INFER_USE_CUDA
#include <cuda_runtime.h>
#endif

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
    if (dims_.empty())
        return 1;  // Scalar has 1 element
    return std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<int64_t>());
}

std::string Shape::to_string() const {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims_.size(); ++i) {
        oss << dims_[i];
        if (i < dims_.size() - 1)
            oss << ", ";
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
// Storage implementation
// ============================================================================

Storage::Storage(size_t capacity_bytes, DeviceType device, size_t alignment) {
    reset(capacity_bytes, device, alignment);
}

Storage::Storage(const std::shared_ptr<void>& external, size_t capacity_bytes, DeviceType device)
    : buffer_(external), capacity_(capacity_bytes), device_(device) {}

void Storage::reset(size_t capacity_bytes, DeviceType device, size_t alignment) {
    if (capacity_bytes == 0) {
        buffer_.reset();
        capacity_ = 0;
        device_ = device;
        return;
    }

    device_ = device;
    size_t aligned_bytes = ((capacity_bytes + alignment - 1) / alignment) * alignment;

    auto allocator_type = device == DeviceType::CPU ? AllocatorFactory::AllocatorType::CPU
                                                    : AllocatorFactory::AllocatorType::CUDA;
    auto allocator = AllocatorFactory::get_allocator(allocator_type);
    if (!allocator) {
        buffer_.reset();
        capacity_ = 0;
        return;
    }

    void* ptr = allocator->allocate(aligned_bytes, alignment);
    if (!ptr) {
        buffer_.reset();
        capacity_ = 0;
        return;
    }

    buffer_.reset(ptr, [allocator](void* p) {
        allocator->deallocate(p);
    });

    // Zero-initialize memory based on device type
#ifdef MINI_INFER_USE_CUDA
    if (device == DeviceType::CUDA) {
        cudaError_t status = cudaMemset(ptr, 0, aligned_bytes);
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[Tensor] cudaMemset failed: " + std::string(cudaGetErrorString(status)));
        }
    } else
#endif
    {
        std::memset(ptr, 0, aligned_bytes);
    }
    capacity_ = aligned_bytes;
}

void Storage::set_external(const std::shared_ptr<void>& external, size_t capacity_bytes,
                           DeviceType device) {
    buffer_ = external;
    capacity_ = capacity_bytes;
    device_ = device;
}

// ============================================================================
// Tensor implementation
// ============================================================================

Tensor::Tensor(const Shape& shape, DataType dtype, DeviceType device, size_t alignment)
    : shape_(shape), dtype_(dtype), device_(device), alignment_(alignment) {
    allocate();
}

void* Tensor::data() {
    if (!storage_ || storage_->empty()) {
        return nullptr;
    }
    auto base = static_cast<uint8_t*>(storage_->data());
    return base ? base + storage_offset_ : nullptr;
}

const void* Tensor::data() const {
    if (!storage_ || storage_->empty()) {
        return nullptr;
    }
    auto base = static_cast<const uint8_t*>(storage_->data());
    return base ? base + storage_offset_ : nullptr;
}

size_t Tensor::element_size() const {
    switch (dtype_) {
        case DataType::FLOAT32:
            return sizeof(float);
        case DataType::FLOAT16:
            return 2;
        case DataType::INT32:
            return sizeof(int32_t);
        case DataType::INT64:
            return sizeof(int64_t);
        case DataType::INT8:
            return sizeof(int8_t);
        case DataType::UINT8:
            return sizeof(uint8_t);
        case DataType::BOOL:
            return sizeof(bool);
        default:
            return 0;
    }
}

size_t Tensor::size_in_bytes() const {
    return static_cast<size_t>(shape_.numel()) * element_size();
}

void Tensor::allocate() {
    size_t bytes = size_in_bytes();
    if (bytes == 0) {
        storage_.reset();
        storage_offset_ = 0;
        strides_.clear();
        return;
    }

    storage_ = std::make_shared<Storage>(bytes, device_, alignment_);
    storage_offset_ = 0;
    compute_contiguous_strides();
}

void Tensor::reshape(const Shape& new_shape) {
    if (new_shape.numel() != shape_.numel()) {
        return;
    }
    shape_ = new_shape;
    compute_contiguous_strides();
}

void Tensor::ensure_contiguous_storage(size_t new_size_bytes) {
    if (new_size_bytes == 0) {
        storage_.reset();
        storage_offset_ = 0;
        return;
    }

    size_t available = 0;
    bool unique_storage = false;
    if (storage_) {
        available = storage_->capacity() > storage_offset_
                        ? storage_->capacity() - storage_offset_
                        : 0;
        unique_storage = (storage_.use_count() == 1 && storage_offset_ == 0);
    }

    if (storage_ && unique_storage && new_size_bytes <= available) {
        return;
    }

    auto new_storage = std::make_shared<Storage>(new_size_bytes, device_, alignment_);
    if (storage_ && storage_->data()) {
        size_t copy_size = std::min(size_in_bytes(), new_size_bytes);
        if (copy_size > 0) {
#ifdef MINI_INFER_USE_CUDA
            if (device_ == DeviceType::CUDA) {
                cudaError_t status = cudaMemcpy(new_storage->data(), data(), copy_size, cudaMemcpyDeviceToDevice);
                if (status != cudaSuccess) {
                    MI_LOG_ERROR("[Tensor] cudaMemcpy failed: " + std::string(cudaGetErrorString(status)));
                }
            } else
#endif
            {
                std::memcpy(new_storage->data(), data(), copy_size);
            }
        }
    }

    storage_ = new_storage;
    storage_offset_ = 0;
}

void Tensor::resize(const Shape& new_shape) {
    size_t new_size = static_cast<size_t>(new_shape.numel()) * element_size();
    size_t old_size = size_in_bytes();

    ensure_contiguous_storage(new_size);
    shape_ = new_shape;
    compute_contiguous_strides();

    if (storage_ && storage_->data() && new_size > old_size) {
        auto ptr = static_cast<uint8_t*>(data());
        if (ptr) {
#ifdef MINI_INFER_USE_CUDA
            if (device_ == DeviceType::CUDA) {
                cudaError_t status = cudaMemset(ptr + old_size, 0, new_size - old_size);
                if (status != cudaSuccess) {
                    MI_LOG_ERROR("[Tensor] cudaMemset failed: " + std::string(cudaGetErrorString(status)));
                }
            } else
#endif
            {
                std::memset(ptr + old_size, 0, new_size - old_size);
            }
        }
    }
}

std::shared_ptr<Tensor> Tensor::view(const Shape& new_shape) const {
    if (new_shape.numel() != shape_.numel()) {
        return nullptr;
    }

    auto view_tensor = std::make_shared<Tensor>();
    view_tensor->shape_ = new_shape;
    view_tensor->dtype_ = dtype_;
    view_tensor->storage_ = storage_;
    view_tensor->storage_offset_ = storage_offset_;
    view_tensor->device_ = device_;
    view_tensor->alignment_ = alignment_;
    view_tensor->compute_contiguous_strides();
    return view_tensor;
}

std::shared_ptr<Tensor> Tensor::create(const Shape& shape, DataType dtype, DeviceType device) {
    return std::make_shared<Tensor>(shape, dtype, device);
}

void Tensor::bind_external_data(const std::shared_ptr<void>& data, size_t capacity_bytes,
                                DeviceType device) {
    if (!storage_) {
        storage_ = std::make_shared<Storage>(data, capacity_bytes, device);
    } else {
        storage_->set_external(data, capacity_bytes, device);
    }
    storage_offset_ = 0;
    device_ = device;
    compute_contiguous_strides();
}

bool Tensor::bind_external_data_with_offset(const std::shared_ptr<void>& data,
                                            size_t capacity_bytes, size_t offset_bytes,
                                            DeviceType device) {
    const size_t required = size_in_bytes();
    if (offset_bytes + required > capacity_bytes) {
        return false;
    }

    if (!storage_) {
        storage_ = std::make_shared<Storage>(data, capacity_bytes, device);
    } else {
        storage_->set_external(data, capacity_bytes, device);
    }
    storage_offset_ = offset_bytes;
    device_ = device;
    compute_contiguous_strides();
    return true;
}

void Tensor::set_shape_metadata(const Shape& shape) {
    shape_ = shape;
    compute_contiguous_strides();
}

void Tensor::compute_contiguous_strides() {
    strides_.assign(shape_.ndim(), 0);
    if (shape_.ndim() == 0) {
        return;
    }

    int64_t stride = 1;
    for (int64_t idx = static_cast<int64_t>(shape_.ndim()) - 1; idx >= 0; --idx) {
        strides_[static_cast<size_t>(idx)] = stride;
        stride *= shape_[static_cast<size_t>(idx)];
    }
}

}  // namespace core
}  // namespace mini_infer
