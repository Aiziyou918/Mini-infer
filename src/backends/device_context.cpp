#include "mini_infer/backends/device_context.h"

namespace mini_infer {
namespace backends {

core::DeviceType CPUDeviceContext::device_type() const {
    return core::DeviceType::CPU;
}

void CPUDeviceContext::synchronize() {
    // CPU backend is synchronous by default.
}

}  // namespace backends
}  // namespace mini_infer
