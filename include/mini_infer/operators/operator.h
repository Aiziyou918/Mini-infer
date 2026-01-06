#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mini_infer/core/op_type.h"
#include "mini_infer/core/tensor.h"
#include "mini_infer/core/types.h"
#include "mini_infer/operators/plugin_base.h"

namespace mini_infer {
namespace operators {

/**
 * @brief Operator parameter base class
 */
struct OpParam {
    virtual ~OpParam() = default;
};

/**
 * @brief Operator base class
 *
 * This class serves as a metadata container for graph nodes.
 * All actual computation is handled by the Plugin system via cached_plugin_.
 *
 * The Operator class is kept for:
 * - Graph construction and node management
 * - Parameter storage
 * - Plugin caching
 */
class Operator {
   public:
    explicit Operator(const std::string& name)
        : name_(name), op_type_(core::string_to_op_type(name)) {}

    explicit Operator(const std::string& name, core::OpType op_type)
        : name_(name), op_type_(op_type) {}

    virtual ~Operator() = default;

    /**
     * @brief Set the cached plugin for execution
     * @param plugin The plugin to cache (takes ownership)
     */
    void set_cached_plugin(std::unique_ptr<IPlugin> plugin) {
        cached_plugin_ = std::move(plugin);
    }

    /**
     * @brief Get the cached plugin
     * @return Pointer to the cached plugin, or nullptr if not set
     */
    IPlugin* cached_plugin() const {
        return cached_plugin_.get();
    }

    /**
     * @brief Get the name of the operator
     */
    const std::string& name() const {
        return name_;
    }

    /**
     * @brief Get the OpType of the operator
     */
    core::OpType type() const {
        return op_type_;
    }

    /**
     * @brief Set the parameter of the operator
     */
    virtual void set_param(std::shared_ptr<OpParam> param) {
        param_ = param;
    }

    /**
     * @brief Get the parameter of the operator
     */
    std::shared_ptr<OpParam> param() const {
        return param_;
    }

   protected:
    std::string name_;                         ///< The name of the operator
    core::OpType op_type_;                     ///< The OpType of the operator
    std::shared_ptr<OpParam> param_;           ///< The parameter of the operator
    std::unique_ptr<IPlugin> cached_plugin_;   ///< Cached plugin for execution
};

/**
 * @brief Operator factory
 */
class OperatorFactory {
   public:
    using CreateFunc = std::shared_ptr<Operator> (*)();

    /**
     * @brief Register an operator
     * @param op_type The type of the operator to register
     * @param func The function to create the operator
     */
    static void register_operator(const std::string& op_type, CreateFunc func);

    /**
     * @brief Create an operator
     * @param op_type The type of the operator to create
     * @return A shared pointer to the created operator
     */
    static std::shared_ptr<Operator> create_operator(const std::string& op_type);

   private:
    /**
     * @brief Get the registry of the operators
     * @return A reference to the registry of the operators
     */
    static std::unordered_map<std::string, CreateFunc>& get_registry();
};

// Register macro
#define REGISTER_OPERATOR(op_type, op_class)                                 \
    namespace {                                                              \
    std::shared_ptr<Operator> create_##op_class() {                          \
        return std::make_shared<op_class>();                                 \
    }                                                                        \
    struct op_class##_register {                                             \
        op_class##_register() {                                              \
            OperatorFactory::register_operator(#op_type, create_##op_class); \
        }                                                                    \
    };                                                                       \
    static op_class##_register g_##op_class##_register;                      \
    }

}  // namespace operators
}  // namespace mini_infer
