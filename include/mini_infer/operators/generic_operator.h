#pragma once

#include <memory>
#include "mini_infer/operators/operator.h"
#include "mini_infer/operators/plugin_base.h"

namespace mini_infer {
namespace operators {

/**
 * @brief Generic Operator class for use with the Plugin architecture
 *
 * This class is used when importing ONNX models. It stores the operator
 * parameters and relies on the Plugin system for actual computation.
 * The InferencePlan will create and cache the appropriate Plugin based
 * on the OpType and target DeviceType.
 */
class GenericOperator : public Operator {
   public:
    explicit GenericOperator(const std::string& name)
        : Operator(name) {}

    explicit GenericOperator(const std::string& name, core::OpType op_type)
        : Operator(name, op_type) {}

    /**
     * @brief Set the plugin parameter
     * @param param The parameter to set (should match the expected plugin parameter type)
     */
    void set_plugin_param(std::shared_ptr<PluginParam> plugin_param) {
        plugin_param_ = plugin_param;
    }

    /**
     * @brief Get the plugin parameter
     */
    std::shared_ptr<PluginParam> plugin_param() const {
        return plugin_param_;
    }

   private:
    std::shared_ptr<PluginParam> plugin_param_;
};

}  // namespace operators
}  // namespace mini_infer


