#include "mini_infer/backends/cpu_backend.h"
#include "mini_infer/core/allocator.h"
#include <cstring>

namespace mini_infer {
namespace backends {

CPUBackend CPUBackend::instance_;

void* CPUBackend::allocate(size_t size) {
    return core::CPUAllocator::get_instance()->allocate(size);
}

void CPUBackend::deallocate(void* ptr) {
    core::CPUAllocator::get_instance()->deallocate(ptr);
}

void CPUBackend::memcpy(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
}

void CPUBackend::memset(void* ptr, int value, size_t size) {
    std::memset(ptr, value, size);
}

void CPUBackend::copy_tensor(core::Tensor& dst, const core::Tensor& src) {
    if (dst.size_in_bytes() != src.size_in_bytes()) {
        return;
    }
    std::memcpy(dst.data(), src.data(), src.size_in_bytes());
}

CPUBackend* CPUBackend::get_instance() {
    return &instance_;
}

} // namespace backends
} // namespace mini_infer

