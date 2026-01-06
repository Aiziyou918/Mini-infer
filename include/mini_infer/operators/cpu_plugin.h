#pragma once

#include "mini_infer/operators/plugin_base.h"

namespace mini_infer {
namespace operators {

/**
 * @brief CRTP base class for CPU plugins
 *
 * Provides common functionality for CPU-based plugins.
 * Uses Curiously Recurring Template Pattern for static polymorphism.
 *
 * Template parameters:
 *   - Derived: The derived plugin class
 *   - ParamType: The parameter type (default: PluginParam)
 *
 * Usage:
 *   class ReLUCPUPlugin : public CPUPlugin<ReLUCPUPlugin> {
 *       // ...
 *   };
 */
template <typename Derived, typename ParamType = PluginParam>
class CPUPlugin : public IPlugin {
public:
    CPUPlugin() = default;
    ~CPUPlugin() override = default;

    /**
     * @brief Get the device type (always CPU)
     */
    core::DeviceType get_device_type() const noexcept override {
        return core::DeviceType::CPU;
    }

    /**
     * @brief Clone the plugin
     */
    std::unique_ptr<IPlugin> clone() const override {
        auto cloned = std::make_unique<Derived>(static_cast<const Derived&>(*this));
        return cloned;
    }

    /**
     * @brief Get pointer to plugin parameters
     */
    const void* get_param_ptr() const noexcept override {
        return param_ ? param_.get() : nullptr;
    }

    /**
     * @brief Set plugin parameters
     */
    void set_param(std::shared_ptr<PluginParam> param) override {
        param_ = std::dynamic_pointer_cast<ParamType>(param);
    }

    /**
     * @brief Get typed parameters
     */
    const ParamType* param() const {
        return param_.get();
    }

    /**
     * @brief Set typed parameters
     */
    void set_typed_param(const ParamType& param) {
        param_ = std::make_shared<ParamType>(param);
    }

protected:
    std::shared_ptr<ParamType> param_;
};

/**
 * @brief Simple CPU plugin base for operators without parameters
 *
 * Usage:
 *   class ReLUCPUPlugin : public SimpleCPUPlugin<ReLUCPUPlugin> {
 *       // ...
 *   };
 */
template <typename Derived>
class SimpleCPUPlugin : public CPUPlugin<Derived, PluginParam> {
public:
    SimpleCPUPlugin() = default;
    ~SimpleCPUPlugin() override = default;
};

}  // namespace operators
}  // namespace mini_infer
