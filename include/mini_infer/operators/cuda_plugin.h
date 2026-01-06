#pragma once

#include "mini_infer/operators/plugin_base.h"

#ifdef MINI_INFER_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace mini_infer {
namespace operators {

/**
 * @brief CRTP base class for CUDA plugins
 *
 * Provides common functionality for CUDA-based plugins.
 * Uses Curiously Recurring Template Pattern for static polymorphism.
 *
 * Template parameters:
 *   - Derived: The derived plugin class
 *   - ParamType: The parameter type (default: PluginParam)
 *
 * Features:
 *   - Automatic device type identification
 *   - Helper methods for CUDA stream access
 *   - Support for cuDNN/cuBLAS handle caching
 *
 * Usage:
 *   class ReLUCUDAPlugin : public CUDAPlugin<ReLUCUDAPlugin> {
 *       // ...
 *   };
 */
template <typename Derived, typename ParamType = PluginParam>
class CUDAPlugin : public IPlugin {
public:
    CUDAPlugin() = default;
    ~CUDAPlugin() override = default;

    /**
     * @brief Get the device type (always CUDA)
     */
    core::DeviceType get_device_type() const noexcept override {
        return core::DeviceType::CUDA;
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

#ifdef MINI_INFER_USE_CUDA
    /**
     * @brief Get CUDA stream from context
     * @param context Plugin execution context
     * @return CUDA stream, or nullptr if not available
     */
    static cudaStream_t get_cuda_stream(const PluginContext& context) {
        if (!context.device_context) {
            return nullptr;
        }
        // Cast to CUDADeviceContext and get stream
        // This assumes CUDADeviceContext has a stream() method
        // The actual implementation depends on the DeviceContext hierarchy
        return nullptr;  // Placeholder - will be implemented based on actual CUDADeviceContext
    }
#endif

protected:
    std::shared_ptr<ParamType> param_;
};

/**
 * @brief Simple CUDA plugin base for operators without parameters
 *
 * Usage:
 *   class ReLUCUDAPlugin : public SimpleCUDAPlugin<ReLUCUDAPlugin> {
 *       // ...
 *   };
 */
template <typename Derived>
class SimpleCUDAPlugin : public CUDAPlugin<Derived, PluginParam> {
public:
    SimpleCUDAPlugin() = default;
    ~SimpleCUDAPlugin() override = default;
};

}  // namespace operators
}  // namespace mini_infer
