#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
 * Key concepts:
 * - Shape tensors: Tensors that represent shapes (e.g., output of Shape op)
 * - Shape values: Logical values (vector<int64_t>), not runtime Tensor objects
 * - Evaluation is based on shape_values availability, not t->data()
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
     * @brief Seed shape values from initializers (constants)
     *
     * This populates shape_values_ with values from constant tensors
     * that are known at build time (e.g., axes, starts, ends for Slice).
     */
    void seed_from_initializers();

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
        const std::unordered_map<std::string, core::Shape>& input_shapes);

    /**
     * @brief Provide known output shapes for non-input nodes (runtime shape inference)
     *
     * When set, shape evaluation can use these shapes instead of stale output_tensors.
     */
    void set_known_output_shapes(
        const std::vector<std::vector<core::Shape>>* shapes_by_id);

    /**
     * @brief Get the evaluated shape value for a tensor
     * @param tensor_name Name of the tensor
     * @return Pointer to ShapeValue if available, nullptr otherwise
     */
    const ShapeValue* get_shape_value(const std::string& tensor_name) const;

    /**
     * @brief Get the evaluated shape value by node ID
     * @param node_id Node ID
     * @return Pointer to ShapeValue if available, nullptr otherwise
     */
    const ShapeValue* get_shape_value(size_t node_id) const;

    /**
     * @brief Check if a tensor is a shape tensor
     * @param tensor_name Name of the tensor
     * @return True if the tensor is a shape tensor
     */
    bool is_shape_tensor(const std::string& tensor_name) const;

    /**
     * @brief Get all evaluated shape values
     * @return Map of tensor names to shape values
     */
    const std::unordered_map<std::string, ShapeValue>& get_all_shape_values() const {
        return shape_values_;
    }

    /**
     * @brief Clear all evaluated shape values
     */
    void clear();

private:
    /**
     * @brief Check if a node produces shape tensors
     */
    bool is_shape_tensor_op(core::OpType op_type) const;

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
     * @return The shape if available
     */
    core::Shape get_input_shape(
        const std::shared_ptr<graph::Node>& node, int port) const;

    std::shared_ptr<graph::Graph> graph_;
    std::vector<std::shared_ptr<graph::Node>> sorted_nodes_;

    // Shape values map: tensor_name -> evaluated shape value
    std::unordered_map<std::string, ShapeValue> shape_values_;

    // Shape values by node ID for fast lookup
    std::vector<ShapeValue> shape_values_by_id_;

    // Input shapes from profile
    std::unordered_map<std::string, core::Shape> input_shapes_;

    // Optional runtime shape cache (node_id -> output shapes)
    const std::vector<std::vector<core::Shape>>* known_output_shapes_by_id_{nullptr};

    // Set of shape tensor names (for quick lookup)
    std::unordered_set<std::string> shape_tensor_names_;
};

}  // namespace runtime
}  // namespace mini_infer
