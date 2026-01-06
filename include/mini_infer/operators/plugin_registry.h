#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "mini_infer/core/op_type.h"
#include "mini_infer/core/types.h"
#include "mini_infer/operators/plugin_base.h"

namespace mini_infer {
namespace operators {

/**
 * @brief Plugin registry key
 *
 * Uses binary tuple (OpType, DeviceType) for registration.
 * DataType is handled at runtime by the plugin itself.
 */
struct PluginKey {
    core::OpType op_type;
    core::DeviceType device_type;

    bool operator==(const PluginKey& other) const {
        return op_type == other.op_type && device_type == other.device_type;
    }
};

/**
 * @brief Hash function for PluginKey
 */
struct PluginKeyHash {
    size_t operator()(const PluginKey& key) const {
        const auto op_hash = std::hash<int>{}(static_cast<int>(key.op_type));
        const auto dev_hash = std::hash<int>{}(static_cast<int>(key.device_type));
        size_t seed = op_hash;
        seed ^= dev_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

/**
 * @brief Unified plugin registry
 *
 * Singleton registry for all plugins. Replaces both OperatorFactory and KernelRegistry.
 * Registration key is (OpType, DeviceType).
 */
class PluginRegistry {
public:
    /**
     * @brief Get the singleton instance
     * @return Reference to the registry
     */
    static PluginRegistry& instance() {
        static PluginRegistry registry;
        return registry;
    }

    /**
     * @brief Register a plugin creator
     * @param creator Unique pointer to the creator
     */
    void register_creator(std::unique_ptr<IPluginCreator> creator) {
        if (!creator) {
            return;
        }

        PluginKey key{creator->get_op_type(), creator->get_device_type()};
        std::lock_guard<std::mutex> lock(mutex_);
        creators_[key] = std::move(creator);
    }

    /**
     * @brief Create a plugin by OpType and DeviceType
     * @param op_type Operator type
     * @param device_type Device type
     * @return Unique pointer to the created plugin, or nullptr if not found
     */
    std::unique_ptr<IPlugin> create_plugin(
        core::OpType op_type,
        core::DeviceType device_type) const {
        PluginKey key{op_type, device_type};
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = creators_.find(key);
        if (it == creators_.end()) {
            return nullptr;
        }
        return it->second->create_plugin();
    }

    /**
     * @brief Create a plugin by type name and DeviceType
     * @param type_name Operator type name string
     * @param device_type Device type
     * @return Unique pointer to the created plugin, or nullptr if not found
     */
    std::unique_ptr<IPlugin> create_plugin(
        const std::string& type_name,
        core::DeviceType device_type) const {
        core::OpType op_type = core::string_to_op_type(type_name);
        return create_plugin(op_type, device_type);
    }

    /**
     * @brief Check if a plugin is registered
     * @param op_type Operator type
     * @param device_type Device type
     * @return true if registered, false otherwise
     */
    bool has_plugin(core::OpType op_type, core::DeviceType device_type) const {
        PluginKey key{op_type, device_type};
        std::lock_guard<std::mutex> lock(mutex_);
        return creators_.find(key) != creators_.end();
    }

    /**
     * @brief Get all registered plugin keys
     * @return Vector of registered keys
     */
    std::vector<PluginKey> get_registered_keys() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<PluginKey> keys;
        keys.reserve(creators_.size());
        for (const auto& pair : creators_) {
            keys.push_back(pair.first);
        }
        return keys;
    }

private:
    PluginRegistry() = default;
    PluginRegistry(const PluginRegistry&) = delete;
    PluginRegistry& operator=(const PluginRegistry&) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<PluginKey, std::unique_ptr<IPluginCreator>, PluginKeyHash> creators_;
};

/**
 * @brief Helper class for static plugin registration
 */
template <typename CreatorType>
class PluginRegistrar {
public:
    PluginRegistrar() {
        PluginRegistry::instance().register_creator(std::make_unique<CreatorType>());
    }
};

/**
 * @brief Macro to register a plugin
 *
 * Usage:
 *   REGISTER_PLUGIN(ReLUCPUPlugin, ReLUCPUPluginCreator)
 */
#define REGISTER_PLUGIN(plugin_class, creator_class) \
    namespace {                                      \
    static ::mini_infer::operators::PluginRegistrar<creator_class> \
        g_##plugin_class##_registrar;                \
    }

/**
 * @brief Macro to define a simple plugin creator
 *
 * Usage:
 *   DEFINE_PLUGIN_CREATOR(ReLUCPUPlugin, "Relu", kRELU, CPU)
 */
#define DEFINE_PLUGIN_CREATOR(plugin_class, type_name, op_type_enum, device_type_enum) \
    class plugin_class##Creator : public ::mini_infer::operators::IPluginCreator {     \
    public:                                                                            \
        const char* get_plugin_type() const noexcept override { return type_name; }    \
        ::mini_infer::core::OpType get_op_type() const noexcept override {             \
            return ::mini_infer::core::OpType::op_type_enum;                           \
        }                                                                              \
        ::mini_infer::core::DeviceType get_device_type() const noexcept override {     \
            return ::mini_infer::core::DeviceType::device_type_enum;                   \
        }                                                                              \
        std::unique_ptr<::mini_infer::operators::IPlugin> create_plugin() const override { \
            return std::make_unique<plugin_class>();                                   \
        }                                                                              \
    };

/**
 * @brief Combined macro to define creator and register plugin
 *
 * Usage:
 *   REGISTER_PLUGIN_SIMPLE(ReLUCPUPlugin, "Relu", kRELU, CPU)
 */
#define REGISTER_PLUGIN_SIMPLE(plugin_class, type_name, op_type_enum, device_type_enum) \
    DEFINE_PLUGIN_CREATOR(plugin_class, type_name, op_type_enum, device_type_enum)      \
    REGISTER_PLUGIN(plugin_class, plugin_class##Creator)

}  // namespace operators
}  // namespace mini_infer
