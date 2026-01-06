#pragma once

#include "mini_infer/operators/operator.h"
#include "mini_infer/operators/plugin_base.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/kernels/kernel_base.h"

namespace mini_infer {
namespace operators {

/**
 * @brief Adapter class that wraps IPlugin to work with the existing Operator interface
 *
 * This adapter allows the new Plugin system to be used with the existing Runtime
 * infrastructure without requiring immediate changes to Node, ExecutionContext, etc.
 *
 * Usage:
 *   auto plugin = PluginRegistry::instance().create_plugin(op_type, device_type);
 *   auto op = std::make_shared<PluginOperatorAdapter>(std::move(plugin));
 *   node->set_operator(op);
 */
class PluginOperatorAdapter : public Operator {
public:
    explicit PluginOperatorAdapter(std::unique_ptr<IPlugin> plugin)
        : Operator(plugin ? plugin->get_plugin_type() : "Unknown",
                   plugin ? plugin->get_op_type() : core::OpType::kUNKNOWN)
        , plugin_(std::move(plugin)) {}

    ~PluginOperatorAdapter() override = default;

    /**
     * @brief Forward inference using the wrapped Plugin
     */
    core::Status forward(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs) override {

        if (!plugin_) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        // Create plugin context
        PluginContext context;
        context.device_context = kernels::get_current_device_context();
        context.workspace = nullptr;
        context.workspace_size = 0;

        return plugin_->enqueue(inputs, outputs, context);
    }

    /**
     * @brief Infer output shapes using the wrapped Plugin
     */
    core::Status infer_shape(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) override {

        if (!plugin_) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        return plugin_->infer_output_shapes(input_shapes, output_shapes);
    }

    /**
     * @brief Get the wrapped Plugin
     */
    IPlugin* get_plugin() const {
        return plugin_.get();
    }

    /**
     * @brief Get the wrapped Plugin (mutable)
     */
    IPlugin* mutable_plugin() {
        return plugin_.get();
    }

    /**
     * @brief Check if this adapter has a valid plugin
     */
    bool has_plugin() const {
        return plugin_ != nullptr;
    }

    /**
     * @brief Get the device type of the wrapped Plugin
     */
    core::DeviceType device_type() const {
        return plugin_ ? plugin_->get_device_type() : core::DeviceType::CPU;
    }

private:
    std::unique_ptr<IPlugin> plugin_;
};

/**
 * @brief Helper function to create a PluginOpetorAdapter from PluginRegistry
 *
 * @param op_type The operator type
 * @param device_type The device type (CPU or CUDA)
 * @return A shared pointer to the adapter, or nullptr if plugin not found
 */
inline std::shared_ptr<PluginOperatorAdapter> create_plugin_operator(
    core::OpType op_type,
    core::DeviceType device_type) {

    auto plugin = PluginRegistry::instance().create_plugin(op_type, device_type);
    if (!plugin) {
        return nullptr;
    }
    return std::make_shared<PluginOperatorAdapter>(std::move(plugin));
}

/**
 * @brief Helper function to create a PluginOperatorAdapter from PluginRegistry by name
 *
 * @param type_name The plugin type name (e.g., "Relu", "Conv")
 * @param device_type The device type (CPU or CUDA)
 * @return A shared pointer to the adapter, or nullptr if plugin not found
 */
inline std::shared_ptr<PluginOperatorAdapter> create_plugin_operator(
    const std::string& type_name,
    core::DeviceType device_type) {

    auto plugin = PluginRegistry::instance().create_plugin(type_name, device_type);
    if (!plugin) {
        return nullptr;
    }
    return std::make_shared<PluginOperatorAdapter>(std::move(plugin));
}

}  // namespace operators
}  // namespace mini_infer
