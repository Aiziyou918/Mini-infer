#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mini_infer/core/op_type.h"
#include "mini_infer/core/tensor.h"
#include "mini_infer/core/types.h"


namespace mini_infer {
namespace operators {

/**
 * @brief Operator parameter base class
 */
struct OpParam {
    virtual ~OpParam() = default;
};

/**
 * @brief Operator interface abstract class
 */
class Operator {
   public:
    explicit Operator(const std::string& name)
        : name_(name), op_type_(core::string_to_op_type(name)) {}

    explicit Operator(const std::string& name, core::OpType op_type)
        : name_(name), op_type_(op_type) {}

    virtual ~Operator() = default;

    /*
     * @brief Forward inference
     * @param inputs The input tensors
     * @param outputs The output tensors
     * @return The status of the forward inference
     */
    virtual core::Status forward(const std::vector<std::shared_ptr<core::Tensor>>& inputs,
                                 std::vector<std::shared_ptr<core::Tensor>>& outputs) = 0;

    /**
     * @brief Infer the output shape
     * @param input_shapes The input shapes
     * @param output_shapes The output shapes
     * @return The status of the infer shape
     */
    virtual core::Status infer_shape(const std::vector<core::Shape>& input_shapes,
                                     std::vector<core::Shape>& output_shapes) = 0;

    /**
     * @brief Get the name of the operator
     * @return The name of the operator
     */
    const std::string& name() const {
        return name_;
    }

    /**
     * @brief Get the OpType of the operator
     * @return The OpType of the operator
     */
    core::OpType type() const {
        return op_type_;
    }

    /**
     * @brief Set the parameter of the operator
     * @param param The parameter of the operator
     */
    virtual void set_param(std::shared_ptr<OpParam> param) {
        param_ = param;
    }

   protected:
    std::string name_;                //< The name of the operator
    core::OpType op_type_;            //< The OpType of the operator (cached for fast access)
    std::shared_ptr<OpParam> param_;  //< The parameter of the operator
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
