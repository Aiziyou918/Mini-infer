#pragma once

#include <unordered_set>
#include <vector>

#include "mini_infer/graph/graph_optimizer.h"
#include "mini_infer/graph/node.h"
#include "mini_infer/core/tensor.h"

namespace mini_infer {
namespace graph {

/**
 * @brief Constant Folding Pass (TensorRT-style)
 *
 * Folds constant subgraphs at build time to reduce runtime overhead.
 * Similar to TensorRT's constant folding optimization.
 *
 * Algorithm:
 * 1. Collect all initializers and Constant nodes into Graph::constants_
 * 2. Topological traversal: for each node
 *    - Check if all inputs are constants
 *    - If yes, execute on CPU and register output as constant
 *    - Mark node for deletion
 * 3. Remove folded nodes from graph
 *
 * Supported operators (Phase 1):
 * - Shape, Gather, Unsqueeze, Concat, Cast, Reshape, Slice, ConstantOfShape
 * - Any operator with CPU plugin implementation
 *
 * Benefits:
 * - Reduces runtime operator count
 * - Simplifies shape inference (especially for BERT-style models)
 * - Enables further optimizations
 */
class ConstantFoldingPass : public OptimizationPass {
   public:
    ConstantFoldingPass();
    ~ConstantFoldingPass() override = default;

    /**
     * @brief Apply constant folding optimization
     * @param graph Graph to optimize
     * @param num_modifications Output: number of nodes folded
     * @return Status code
     */
    core::Status apply(Graph* graph, int& num_modifications) override;

   private:
    /**
     * @brief Check if a node can be folded (all inputs are constants)
     * @param node Node to check
     * @param graph Graph containing the node
     * @return true if node can be folded
     */
    bool can_fold_node(const std::shared_ptr<Node>& node, Graph* graph) const;

    /**
     * @brief Execute a node on CPU and return output tensors
     * @param node Node to execute
     * @param graph Graph containing the node
     * @param output_tensors Output: computed constant tensors
     * @return Status code
     */
    core::Status execute_node_on_cpu(const std::shared_ptr<Node>& node, Graph* graph,
                                     std::vector<std::shared_ptr<core::Tensor>>& output_tensors);

    /**
     * @brief Collect initializers into Graph::constants_
     * @param graph Graph to process
     */
    void collect_initializers(Graph* graph);

    /**
     * @brief Get input tensors for a node (from constants or predecessor outputs)
     * @param node Node to get inputs for
     * @param graph Graph containing the node
     * @param input_tensors Output: input tensors
     * @return true if all inputs are available
     */
    bool get_node_inputs(const std::shared_ptr<Node>& node, Graph* graph,
                        std::vector<std::shared_ptr<core::Tensor>>& input_tensors) const;
};

}  // namespace graph
}  // namespace mini_infer
