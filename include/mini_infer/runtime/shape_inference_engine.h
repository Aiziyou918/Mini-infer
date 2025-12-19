#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "mini_infer/core/tensor.h"
#include "mini_infer/core/types.h"
#include "mini_infer/graph/graph.h"


namespace mini_infer {
namespace runtime {

/**
 * @brief Runtime Shape Inference Engine (TensorRT-style)
 *
 * Performs shape inference at runtime when input shapes change.
 * Similar to TensorRT's IExecutionContext::setInputShape() + enqueueV3() flow.
 *
 * Key features:
 * - Re-infer shapes when inputs change
 * - Cache inference results for performance
 * - Validate shapes against optimization profile
 * - Return which tensors need reallocation
 */
class ShapeInferenceEngine {
   public:
    explicit ShapeInferenceEngine(std::shared_ptr<graph::Graph> graph);
    ~ShapeInferenceEngine() = default;

    struct RuntimeInputShape {
        size_t node_id{0};
        core::Shape shape;
    };

    /**
     * @brief Infer all tensor shapes based on current input shapes
     *
     * Performs shape propagation through the graph in topological order.
     *
     * @param input_shapes Map of input name to current shape
     * @return Status::SUCCESS if inference succeeded
     */
    core::Status infer_shapes(const std::unordered_map<std::string, core::Shape>& input_shapes);
    core::Status infer_shapes(const std::vector<RuntimeInputShape>& input_shapes);

    /**
     * @brief Get inferred shape for a tensor
     *
     * @param tensor_name Name of the tensor
     * @return Pointer to inferred shape, or nullptr if not found
     */
    const core::Shape* get_inferred_shape(const std::string& tensor_name) const;

    /**
     * @brief Check if shapes have changed since last inference
     *
     * @param input_shapes New input shapes to check
     * @return True if shapes are different from cached shapes
     */
    bool shapes_changed(const std::unordered_map<std::string, core::Shape>& input_shapes) const;
    bool shapes_changed(const std::vector<RuntimeInputShape>& input_shapes) const;

    /**
     * @brief Get list of tensors that need reallocation
     *
     * Compares current tensor allocations with inferred shapes.
     *
     * @return Vector of tensor names that need reallocation
     */
    std::vector<std::string> get_tensors_needing_reallocation() const;

    /**
     * @brief Clear cached inference results
     */
    void clear_cache();

    /**
     * @brief Enable/disable verbose logging
     */
    void set_verbose(bool verbose) {
        verbose_ = verbose;
    }

   private:
    std::shared_ptr<graph::Graph> graph_;
    std::vector<std::shared_ptr<graph::Node>> sorted_nodes_;

    // Cached inference results (indexed by node ID for O(1) access)
    // Each node can have multiple outputs, so we store vector<Shape> per node
    std::vector<std::vector<core::Shape>> inferred_shapes_;           // [node_id][output_index]
    std::unordered_map<std::string, core::Shape> last_input_shapes_lookup_;
    std::vector<RuntimeInputShape> last_input_shapes_;

    bool verbose_{false};

    /**
     * @brief Perform topological sort if needed
     */
    core::Status ensure_sorted();
    core::Status infer_shapes_internal(const std::vector<RuntimeInputShape>& input_shapes);
};

}  // namespace runtime
}  // namespace mini_infer
