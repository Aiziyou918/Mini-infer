#pragma once

#include <unordered_set>

#include "mini_infer/graph/graph_optimizer.h"
#include "mini_infer/graph/node.h"

namespace mini_infer {
namespace graph {

/**
 * @brief Operator Fusion Pass (TensorRT-style)
 *
 * Aligns 100% with TensorRT's layer fusion optimization.
 *
 * TensorRT approach:
 * - Does NOT create separate fused layers (e.g., "ConvReLU")
 * - Sets activation directly on Conv layer: conv->setActivation(type)
 * - Uses dedicated fusion functions, not generic pattern matching
 *
 * Supported fusions:
 * - Conv2D + Activation (ReLU, Sigmoid, Tanh, LeakyReLU, ELU)
 * - (Future) Conv2D + BatchNorm
 * - (Future) Conv2D + BatchNorm + Activation
 */
class FusionPass : public OptimizationPass {
   public:
    FusionPass();
    ~FusionPass() override = default;

    /**
     * @brief Apply fusion optimization
     * @param graph Graph to optimize
     * @param num_modifications Output: number of fusions performed
     * @return Status code
     */
    core::Status apply(Graph* graph, int& num_modifications) override;

   private:
    /**
     * @brief Try to fuse Conv2D + Activation (TensorRT-style)
     *
     * Checks:
     * - Conv has exactly one consumer
     * - Consumer is a supported activation
     * - Not already marked for deletion
     *
     * If successful:
     * - Sets activation on Conv2D
     * - Marks activation node for deletion
     * - Reconnects graph structure
     *
     * @param graph Graph to modify
     * @param conv_node Conv2D node to check
     * @param nodes_to_delete Set to collect nodes marked for deletion
     * @return true if fusion was performed
     */
    bool try_fuse_conv_activation(Graph* graph, std::shared_ptr<Node> conv_node,
                                  std::unordered_set<std::string>& nodes_to_delete);

    // Future fusion functions:
    // bool try_fuse_conv_bn(Graph* graph, std::shared_ptr<Node> conv_node, ...);
    // bool try_fuse_conv_bn_activation(Graph* graph, std::shared_ptr<Node> conv_node, ...);
    // bool try_fuse_gemm_activation(Graph* graph, std::shared_ptr<Node> gemm_node, ...);
};

}  // namespace graph
}  // namespace mini_infer
