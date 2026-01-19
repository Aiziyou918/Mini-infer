#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mini_infer/core/op_type.h"
#include "mini_infer/core/tensor.h"
#include "mini_infer/core/types.h"
#include "mini_infer/graph/graph.h"

namespace mini_infer {
namespace runtime {

/**
 * @brief Shape Tensor Evaluator - TensorRT-style shape tensor evaluation
 *
 * This class implements a separate pass for evaluating shape tensors,
 * following TensorRT's architecture where shape tensors are evaluated
 * symbolically using a shape_values map, not runtime tensor pointers.
 *
 * Key design principles (TensorRT-style):
 * - Uses node ID as key (not string) for O(1) lookup
 * - Stores node OUTPUT values (not inputs)
 * - Reuses Graph::constants_ from ConstantFoldingPass
 * - Supports both constant and dynamic shape values
 *
 * Typical shape subgraph in BERT/LLM:
 *   Shape -> Gather -> Unsqueeze -> Concat -> Reshape
 */
class ShapeTensorEvaluator {
public:
    /**
     * @brief Shape value type - represents the evaluated value of a shape tensor
     */
    struct ShapeValue {
        std::vector<int64_t> data;      // The actual shape values
        core::DataType dtype{core::DataType::INT64};
        bool is_valid{false};

        ShapeValue() = default;
        explicit ShapeValue(const std::vector<int64_t>& d,
                           core::DataType dt = core::DataType::INT64)
            : data(d), dtype(dt), is_valid(true) {}

        int64_t numel() const { return static_cast<int64_t>(data.size()); }
        bool empty() const { return data.empty(); }
    };

    ShapeTensorEvaluator() = default;
    ~ShapeTensorEvaluator() = default;

    /**
     * @brief Initialize the evaluator with a graph
     * @param graph The computation graph
     */
    void initialize(const std::shared_ptr<graph::Graph>& graph);

    /**
     * @brief Seed shape values from Graph::constants_ (TensorRT-style)
     *
     * Reuses the constant tensors collected by ConstantFoldingPass.
     * Only integer tensors (INT64/INT32) are considered shape tensors.
     */
    void seed_from_graph_constants();

    /**
     * @brief Evaluate all shape tensors in topological order
     *
     * For each shape-tensor-producing op, check if we can evaluate it
     * based on available shape_values (not runtime pointers).
     *
     * @param input_shapes Map of input names to their shapes
     * @return Status code
     */
    core::Status evaluate(
        const std::unordered_map<size_t, core::Shape>& input_shapes);

    /**
     * @brief Incrementally update shape values when a new node shape is inferred
     *
     * TensorRT-style incremental evaluation: only re-evaluate shape tensor nodes
     * that depend on the newly inferred shape.
     *
     * @param node_id The node whose shape was just inferred
     * @param shape The newly inferred shape
     * @return Status code
     */
    core::Status incremental_evaluate(size_t node_id, const core::Shape& shape);

    /**
     * @brief Get the evaluated shape value by node ID
     * @param node_id Node ID
     * @return Pointer to ShapeValue if available, nullptr otherwise
     */
    const ShapeValue* get_shape_value(size_t node_id) const;

    /**
     * @brief Check if a node has a known shape value
     * @param node_id Node ID
     * @return True if the node has a valid shape value
     */
    bool has_shape_value(size_t node_id) const;

    /**
     * @brief Get all evaluated shape values (by node ID)
     * @return Vector of shape values indexed by node ID
     */
    const std::vector<ShapeValue>& get_all_shape_values() const {
        return shape_values_;
    }

    /**
     * @brief Clear all evaluated shape values
     */
    void clear();

    /**
     * @brief Check if an operator type is shape-related
     */
    static bool is_shape_tensor_op(core::OpType op_type);

private:
    /**
     * @brief Check if we can evaluate a shape tensor op
     *
     * This checks if all required inputs have shape values available,
     * NOT if they have runtime data pointers.
     */
    bool can_evaluate(const std::shared_ptr<graph::Node>& node) const;

    /**
     * @brief Evaluate a shape tensor op and store the result
     * @return True if evaluation succeeded
     */
    bool evaluate_node(const std::shared_ptr<graph::Node>& node);

    // Evaluation functions for specific ops
    ShapeValue eval_shape_op(const std::shared_ptr<graph::Node>& node);
    ShapeValue eval_gather_op(const std::shared_ptr<graph::Node>& node);
    ShapeValue eval_unsqueeze_op(const std::shared_ptr<graph::Node>& node);
    ShapeValue eval_squeeze_op(const std::shared_ptr<graph::Node>& node);
    ShapeValue eval_concat_op(const std::shared_ptr<graph::Node>& node);
    ShapeValue eval_cast_op(const std::shared_ptr<graph::Node>& node);
    ShapeValue eval_slice_op(const std::shared_ptr<graph::Node>& node);
    ShapeValue eval_constant_of_shape_op(const std::shared_ptr<graph::Node>& node);
    ShapeValue eval_mul_op(const std::shared_ptr<graph::Node>& node);
    ShapeValue eval_add_op(const std::shared_ptr<graph::Node>& node);
    ShapeValue eval_equal_op(const std::shared_ptr<graph::Node>& node);
    ShapeValue eval_where_op(const std::shared_ptr<graph::Node>& node);
    ShapeValue eval_reshape_op(const std::shared_ptr<graph::Node>& node);

    /**
     * @brief Get input shape value for a node at given port
     * @param node The node
     * @param port Input port index
     * @return Pointer to ShapeValue if available
     */
    const ShapeValue* get_input_shape_value(
        const std::shared_ptr<graph::Node>& node, int port) const;

    /**
     * @brief Get input shape (not shape value) for a node at given port
     * @param node The node
     * @param port Input port index
     * @param input_shapes Map of input node IDs to their shapes
     * @return The shape if available
     */
    core::Shape get_input_shape(
        const std::shared_ptr<graph::Node>& node, int port) const;

    /**
     * @brief Read values from tensor (helper)
     */
    std::vector<int64_t> read_tensor_values(
        const std::shared_ptr<core::Tensor>& tensor) const;

    std::vector<std::vector<size_t>> build_shape_consumers(
        const std::vector<std::shared_ptr<graph::Node>>& sorted_nodes,
        size_t capacity) const;
    int count_unmet_inputs(const std::shared_ptr<graph::Node>& node) const;
    void seed_ready_queue(const std::vector<std::shared_ptr<graph::Node>>& sorted_nodes,
                          size_t capacity,
                          std::vector<int>& pending_inputs,
                          std::deque<size_t>& ready) const;

    std::shared_ptr<graph::Graph> graph_;

    // Shape values indexed by node ID (TensorRT-style)
    // Using node ID for O(1) lookup instead of string
    std::vector<ShapeValue> shape_values_;

    // Input shapes from profile
    std::unordered_map<size_t, core::Shape> input_shapes_;

    // Cached consumers graph for incremental updates
    // consumers_[node_id] = list of shape tensor nodes that depend on node_id
    std::vector<std::vector<size_t>> consumers_;
};

}  // namespace runtime
}  // namespace mini_infer
