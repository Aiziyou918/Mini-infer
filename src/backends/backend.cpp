#include "mini_infer/backends/backend.h"
#include "mini_infer/backends/cpu_backend.h"

namespace mini_infer {
namespace backends {

std::shared_ptr<Backend> BackendFactory::create_backend(core::DeviceType type) {
    switch (type) {
        case core::DeviceType::CPU:
            return std::make_shared<CPUBackend>();
        case core::DeviceType::CUDA:
            // TODO: Implement CUDA backend
            return nullptr;
        default:
            return nullptr;
    }
}

std::shared_ptr<Backend> BackendFactory::get_default_backend() {
    return create_backend(core::DeviceType::CPU);
}

} // namespace backends
} // namespace mini_infer

