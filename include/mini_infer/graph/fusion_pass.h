#pragma once

#include "mini_infer/graph/graph_optimizer.h"
#include "mini_infer/graph/node.h"
#include <functional>
#include <vector>
#include <unordered_set>

namespace mini_infer {
namespace graph {

/**
 * @brief Pattern for operator fusion
 * 
 * Defines a sequence of operators to match and fuse.
 */
struct FusionPattern {
    std::vector<std::string> operator_sequence;  // e.g., ["Conv2D", "ReLU"]
    std::string fused_operator_type;              // e.g., "ConvActivation" (not a real operator, just a type identifier)
    std::string name;                              // Pattern name for logging
    
    // Validation function (optional)
    using ValidatorFunc = std::function<bool(const std::vector<std::shared_ptr<Node>>&)>;
    ValidatorFunc validator = nullptr;
    
    // Note: fused_operator_type is NOT the name of a new operator class.
    // It's just an identifier for the fusion logic in fuse_nodes().
    // TensorRT-style: We set activation on Conv2D, not create new operators.
};

/**
 * @brief Operator Fusion Pass (TensorRT-style)
 * 
 * Aligns 100% with TensorRT's layer fusion optimization.
 * 
 * TensorRT approach:
 * - Does NOT create separate fused layers (e.g., "ConvReLU")
 * - Sets activation directly on Conv layer: conv->setActivation(type)
 * - Supports all activation types
 * 
 * Our approach (identical to TensorRT):
 * - Sets activation on Conv2D: conv->set_activation(type)
 * - Removes activation node from graph
 * - No new operator types created
 * 
 * Supported patterns:
 * - Conv2D + Activation (ReLU, Sigmoid, Tanh, LeakyReLU, ELU, etc.)
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

    /**
     * @brief Add a custom fusion pattern
     * @param pattern Fusion pattern to add
     */
    void add_pattern(const FusionPattern& pattern);

private:
    std::vector<FusionPattern> patterns_;

    /**
     * @brief Find and apply all fusion patterns
     * @param graph Graph to search
     * @param pattern Pattern to match
     * @param num_fused Output: number of fusions
     * @return Status code
     */
    core::Status find_and_fuse(
        Graph* graph, 
        const FusionPattern& pattern,
        int& num_fused
    );

    /**
     * @brief Check if a sequence of nodes matches the pattern
     * @param nodes Nodes to check
     * @param pattern Pattern to match
     * @return true if matches
     */
    bool match_pattern(
        const std::vector<std::shared_ptr<Node>>& nodes,
        const FusionPattern& pattern
    );

    /**
     * @brief Fuse matched nodes into a single node
     * @param graph Graph containing the nodes
     * @param nodes Nodes to fuse (in order)
     * @param pattern Pattern that matched
     * @return Status code
     */
    core::Status fuse_nodes(
        Graph* graph,
        const std::vector<std::shared_ptr<Node>>& nodes,
        const FusionPattern& pattern
    );

    /**
     * @brief Initialize built-in fusion patterns
     */
    void init_builtin_patterns();

    /**
     * @brief Try to fuse Conv2D + Activation (TensorRT-style, optimized)
     * 
     * Fast path fusion for Conv + Activation pattern.
     * Checks:
     * - Conv has exactly one consumer
     * - Consumer is a supported activation
     * - Not already fused
     * 
     * If successful:
     * - Sets activation on Conv2D
     * - Removes activation node
     * - Marks nodes as fused
     * 
     * @param graph Graph to modify
     * @param conv_node Conv2D node to check
     * @param fused_nodes Set of already fused nodes
     * @return true if fusion was performed
     */
    bool try_fuse_conv_activation(
        Graph* graph,
        std::shared_ptr<Node> conv_node,
        std::unordered_set<std::string>& fused_nodes
    );
    
    /**
     * @brief Validator for Conv+Activation pattern (legacy, for pattern-based approach)
     * 
     * @deprecated Use try_fuse_conv_activation() instead for better performance
     * 
     * Validates that:
     * - First node is Conv2D
     * - Second node is a supported activation (ReLU, Sigmoid, Tanh, etc.)
     * - Conv has exactly one consumer (the activation)
     * - Activation consumes Conv's output
     */
    static bool validate_conv_activation(const std::vector<std::shared_ptr<Node>>& nodes);
};

} // namespace graph
} // namespace mini_infer
