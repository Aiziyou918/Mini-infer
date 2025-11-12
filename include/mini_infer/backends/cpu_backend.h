#pragma once

#include "mini_infer/backends/backend.h"

namespace mini_infer {
namespace backends {

/**
 * @brief CPU backend implementation
 */
class CPUBackend : public Backend {
public:
    CPUBackend() = default;
    ~CPUBackend() override = default;
    
    core::DeviceType device_type() const override {
        return core::DeviceType::CPU;
    }
    
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void memcpy(void* dst, const void* src, size_t size) override;
    void memset(void* ptr, int value, size_t size) override;
    
    void copy_tensor(core::Tensor& dst, const core::Tensor& src) override;
    
    void synchronize() override {
        // CPU does not need to synchronize
    }
    
    const char* name() const override {
        return "CPU";
    }
    
    /**
     * @brief Get the instance of the CPU backend
     * @return A pointer to the instance of the CPU backend
     */
    static CPUBackend* get_instance();

private:
    static CPUBackend instance_; //< The instance of the CPU backend
};
} // namespace backends
} // namespace mini_infer
