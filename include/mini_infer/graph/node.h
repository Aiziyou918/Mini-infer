#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mini_infer/operators/operator.h"


namespace mini_infer {
namespace graph {

/**
 * @brief Graph node
 */
class Node {
   public:
    explicit Node(const std::string& name);
    ~Node() = default;

    /**
     * @brief Set the operator
     * @param op The operator to set
     */
    void set_operator(std::shared_ptr<operators::Operator> op) {
        op_ = op;
    }

    /**
     * @brief Get the operator
     * @return The operator
     */
    std::shared_ptr<operators::Operator> get_operator() const {
        return op_;
    }

    /**
     * @brief Add an input node
     * @param input_node The input node to add
     */
    void add_input(std::shared_ptr<Node> input_node);

    /**
     * @brief Add an output node
     * @param output_node The output node to add
     */
    void add_output(std::shared_ptr<Node> output_node);

    /**
     * @brief Get the input nodes
     * @return The input nodes
     */
    const std::vector<std::shared_ptr<Node>>& inputs() const {
        return input_nodes_;
    }
    std::vector<std::shared_ptr<Node>>& mutable_inputs() {
        return input_nodes_;
    }

    /**
     * @brief Get the output nodes
     * @return The output nodes
     */
    const std::vector<std::shared_ptr<Node>>& outputs() const {
        return output_nodes_;
    }
    std::vector<std::shared_ptr<Node>>& mutable_outputs() {
        return output_nodes_;
    }

    /**
     * @brief Get the name of the node
     * @return The name of the node
     */
    const std::string& name() const {
        return name_;
    }

    /**
     * @brief Set the name of the node
     * @param name The name to set
     */
    void set_name(const std::string& name) {
        name_ = name;
    }

    /**
     * @brief Set the input tensors
     * @param tensors The input tensors to set
     */
    void set_input_tensors(const std::vector<std::shared_ptr<core::Tensor>>& tensors) {
        input_tensors_ = tensors;
    }

    /**
     * @brief Set the output tensors
     * @param tensors The output tensors to set
     */
    void set_output_tensors(const std::vector<std::shared_ptr<core::Tensor>>& tensors) {
        output_tensors_ = tensors;
    }

    /**
     * @brief Get the input tensors
     * @return The input tensors
     */
    const std::vector<std::shared_ptr<core::Tensor>>& input_tensors() const {
        return input_tensors_;
    }

    /**
     * @brief Get the output tensors
     * @return The output tensors
     */
    std::vector<std::shared_ptr<core::Tensor>>& output_tensors() {
        return output_tensors_;
    }

   private:
    std::string name_;                         //< The name of the node
    std::shared_ptr<operators::Operator> op_;  //< The operator of the node

    std::vector<std::shared_ptr<Node>> input_nodes_;   //< The input nodes of the node
    std::vector<std::shared_ptr<Node>> output_nodes_;  //< The output nodes of the node

    std::vector<std::shared_ptr<core::Tensor>> input_tensors_;   //< The input tensors of the node
    std::vector<std::shared_ptr<core::Tensor>> output_tensors_;  //< The output tensors of the node
};

}  // namespace graph
}  // namespace mini_infer
