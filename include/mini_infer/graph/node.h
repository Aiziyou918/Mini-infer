#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mini_infer/core/op_type.h"
#include "mini_infer/operators/operator.h"

namespace mini_infer {
namespace graph {

/**
 * @brief Graph node
 *
 * TensorRT-style hybrid architecture:
 * - Caches OpType enum for fast access (graph optimization)
 * - Maintains string name for custom operators
 * - Similar to TensorRT's ILayer::getType() + IPluginV2::getPluginType()
 */
class Node {
   public:
    struct Edge {
        std::shared_ptr<Node> node;
        int src_port{0};
        int dst_port{0};
    };

    explicit Node(const std::string& name);
    ~Node() = default;

    /**
     * @brief Set the operator
     * @param op The operator to set
     *
     * This will automatically cache the OpType for fast access.
     */
    void set_operator(std::shared_ptr<operators::Operator> op);

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
    void add_input(const std::shared_ptr<Node>& input_node, int src_port = 0, int dst_port = 0);

    /**
     * @brief Add an output node
     * @param output_node The output node to add
     */
    void add_output(const std::shared_ptr<Node>& output_node, int src_port = 0, int dst_port = 0);

    /**
     * @brief Get the input nodes
     * @return The input nodes
     */
    const std::vector<Edge>& inputs() const {
        return input_edges_;
    }
    std::vector<Edge>& mutable_inputs() {
        return input_edges_;
    }

    /**
     * @brief Get the output nodes
     * @return The output nodes
     */
    const std::vector<Edge>& outputs() const {
        return output_edges_;
    }
    std::vector<Edge>& mutable_outputs() {
        return output_edges_;
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

    /**
     * @brief Get operator type (fast access for graph optimization)
     * @return Cached OpType enum
     *
     * This is the fast path for graph optimization (switch/case).
     * Similar to TensorRT's ILayer::getType().
     */
    core::OpType type() const {
        return cached_op_type_;
    }

    /**
     * @brief Get operator type name (for custom operators and logging)
     * @return Operator type string
     *
     * This is the slow path for custom operators and debugging.
     * Similar to TensorRT's IPluginV2::getPluginType().
     */
    const char* type_name() const;

    /**
     * @brief Get node ID (assigned during graph build)
     * @return Unique node ID for fast indexing
     */
    size_t id() const {
        return node_id_;
    }

    /**
     * @brief Set node ID (called by Graph during build)
     * @param id Unique node ID
     */
    void set_id(size_t id) {
        node_id_ = id;
    }

   private:
    std::string name_;                         ///< The name of the node
    std::shared_ptr<operators::Operator> op_;  ///< The operator of the node
    core::OpType cached_op_type_;              ///< Cached operator type for fast access
    size_t node_id_{0};                        ///< Unique node ID for fast indexing

    std::vector<Edge> input_edges_;   ///< Incoming edges
    std::vector<Edge> output_edges_;  ///< Outgoing edges

    std::vector<std::shared_ptr<core::Tensor>> input_tensors_;   ///< The input tensors of the node
    std::vector<std::shared_ptr<core::Tensor>> output_tensors_;  ///< The output tensors of the node
};

}  // namespace graph
}  // namespace mini_infer
