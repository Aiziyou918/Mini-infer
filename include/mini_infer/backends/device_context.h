#pragma once

#include "mini_infer/core/types.h"

namespace mini_infer {
namespace backends {

class DeviceContext {
public:
    virtual ~DeviceContext() = default;
    virtual core::DeviceType device_type() const = 0;
    virtual void synchronize() = 0;
};

class CPUDeviceContext : public DeviceContext {
public:
    core::DeviceType device_type() const override;
    void synchronize() override;
};

}  // namespace backends
}  // namespace mini_infer
