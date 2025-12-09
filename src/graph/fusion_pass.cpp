#include "mini_infer/graph/fusion_pass.h"

#include <algorithm>
#include <unordered_set>

#include "mini_infer/operators/activation_type.h"
#include "mini_infer/operators/conv2d.h"
#include "mini_infer/utils/logger.h"

namespace mini_infer {
namespace graph {

namespace {

/**
 * @brief Map activation operator name to ActivationType enum
 *
 * Helper function to convert operator names (e.g., "ReLU") to
 * ActivationType enum values (e.g., ActivationType::RELU).
 *
 * @param act_name Activation operator name
 * @return Corresponding ActivationType, or NONE if unknown
 */
operators::ActivationType map_activation_name_to_type(const std::string& act_name) {
    if (act_name == "ReLU") {
        return operators::ActivationType::RELU;
    } else if (act_name == "Sigmoid") {
        return operators::ActivationType::SIGMOID;
    } else if (act_name == "Tanh") {
        return operators::ActivationType::TANH;
    } else if (act_name == "LeakyReLU") {
        return operators::ActivationType::LEAKY_RELU;
        // TODO: Extract alpha parameter from LeakyReLU operator
    } else if (act_name == "ELU") {
        return operators::ActivationType::ELU;
        // TODO: Extract alpha parameter from ELU operator
    } else if (act_name == "SELU") {
        return operators::ActivationType::SELU;
    } else if (act_name == "Softplus") {
        return operators::ActivationType::SOFTPLUS;
    } else if (act_name == "Softsign") {
        return operators::ActivationType::SOFTSIGN;
    } else if (act_name == "HardSigmoid") {
        return operators::ActivationType::HARD_SIGMOID;
    } else if (act_name == "ThresholdedReLU") {
        return operators::ActivationType::THRESHOLDED_RELU;
    } else {
        return operators::ActivationType::NONE;
    }
}

}  // anonymous namespace

FusionPass::FusionPass() : OptimizationPass("FusionPass") {
    init_builtin_patterns();
}

void FusionPass::init_builtin_patterns() {
    // Conv2D + Activation fusion pattern
    // Supports: ReLU, Sigmoid, Tanh, LeakyReLU, etc.
    FusionPattern conv_activation_pattern;
    conv_activation_pattern.operator_sequence = {"Conv2D", "ReLU"};
    conv_activation_pattern.fused_operator_type = "ConvActivation";
    conv_activation_pattern.name = "Conv+Activation";
    conv_activation_pattern.validator = validate_conv_activation;

    patterns_.push_back(conv_activation_pattern);

    // TODO: Add more patterns
    // - Conv2D + Sigmoid
    // - Conv2D + Tanh
    // - Conv2D + BatchNorm + Activation
}

void FusionPass::add_pattern(const FusionPattern& pattern) {
    patterns_.push_back(pattern);
}

core::Status FusionPass::apply(Graph* graph, int& num_modifications) {
    if (!graph) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    num_modifications = 0;

    /**
     * TensorRT-style Optimization: Deferred Deletion Pattern
     *
     * Design Pattern: Two-Phase Approach
     * ===================================
     *
     * Phase 1: Mark (Traversal + Fusion)
     * - Traverse graph once
     * - Perform fusion operations
     * - Mark nodes for deletion (collect in set)
     * - NO actual deletion during traversal
     *
     * Phase 2: Sweep (Batch Deletion)
     * - After traversal completes
     * - Delete all marked nodes at once
     * - Safe: no iterator invalidation
     *
     * Benefits:
     * [SUCESS] No iterator invalidation (100% safe)
     * [SUCESS] Single traversal (O(N) complexity)
     * [SUCESS] No extra lookups (no get_node calls)
     * [SUCESS] Clear separation of concerns
     * [SUCESS] Aligned with TensorRT/ONNX Runtime design
     *
     * Performance: Optimal for all graph sizes!
     */

    const auto& all_nodes = graph->nodes();
    std::unordered_set<std::string> nodes_to_delete;  // Deferred deletion set

    // ========================================================================
    // Phase 1: Mark - Traverse and fuse (no deletion)
    // ========================================================================
    for (const auto& [name, node] : all_nodes) {
        if (!node || !node->get_operator()) {
            continue;
        }

        // Skip nodes already marked for deletion
        if (nodes_to_delete.count(name) > 0) {
            continue;
        }

        // Check fusion opportunities based on operator type
        const std::string& op_type = node->get_operator()->name();

        if (op_type == "Conv2D") {
            // Try Conv + Activation fusion
            // Note: This marks activation node for deletion, but doesn't delete it
            if (try_fuse_conv_activation(graph, node, nodes_to_delete)) {
                num_modifications++;
            }
            // Future: Add more Conv-related fusions
            // else if (try_fuse_conv_bn_activation(graph, node, nodes_to_delete)) { ... }
            // else if (try_fuse_conv_bn(graph, node, nodes_to_delete)) { ... }
        }
        // Future: Add other operator fusions
        // else if (op_type == "Gemm" || op_type == "MatMul") {
        //     if (try_fuse_gemm_activation(graph, node, nodes_to_delete)) { ... }
        // }
    }

    // ========================================================================
    // Phase 2: Sweep - Batch deletion (safe, no iteration)
    // ========================================================================
    for (const auto& node_name : nodes_to_delete) {
        graph->remove_node(node_name);
    }

    if (num_modifications > 0) {
        MI_LOG_INFO("[FusionPass] Applied " + std::to_string(num_modifications) +
                    " fusion(s), removed " + std::to_string(nodes_to_delete.size()) + " node(s)");
    }

    return core::Status::SUCCESS;
}

core::Status FusionPass::find_and_fuse(Graph* graph, const FusionPattern& pattern, int& num_fused) {
    num_fused = 0;

    const auto& all_nodes = graph->nodes();
    if (all_nodes.empty()) {
        return core::Status::SUCCESS;
    }

    size_t pattern_length = pattern.operator_sequence.size();
    if (pattern_length == 0) {
        return core::Status::SUCCESS;
    }

    std::unordered_set<std::string> nodes_to_delete;  // Deferred deletion set

    // Collect all nodes first to avoid iterator invalidation
    std::vector<std::shared_ptr<Node>> nodes_snapshot;
    nodes_snapshot.reserve(all_nodes.size());
    for (const auto& node_entry : all_nodes) {
        if (node_entry.second && node_entry.second->get_operator()) {
            nodes_snapshot.push_back(node_entry.second);
        }
    }

    // Search for pattern matches
    for (const auto& node : nodes_snapshot) {
        if (!node || !node->get_operator()) {
            continue;
        }

        // Skip if already marked for deletion
        if (nodes_to_delete.count(node->name()) > 0) {
            continue;
        }

        // Check if this node starts a matching pattern
        if (node->get_operator()->name() != pattern.operator_sequence[0]) {
            continue;
        }

        // Collect candidate nodes for this pattern
        std::vector<std::shared_ptr<Node>> candidate_nodes;
        candidate_nodes.push_back(node);

        // Try to extend the pattern
        auto current_node = node;
        for (size_t i = 1; i < pattern_length; ++i) {
            // Get output consumers of current node
            const auto& consumers = current_node->outputs();

            // Pattern must have single consumer path
            if (consumers.size() != 1) {
                break;
            }

            auto next_node = consumers[0];

            if (!next_node || !next_node->get_operator()) {
                break;
            }

            // Check if next node matches pattern
            if (next_node->get_operator()->name() != pattern.operator_sequence[i]) {
                break;
            }

            // Check if already marked for deletion
            if (nodes_to_delete.count(next_node->name()) > 0) {
                break;
            }

            candidate_nodes.push_back(next_node);
            current_node = next_node;
        }

        // Check if we found a complete pattern
        if (candidate_nodes.size() == pattern_length) {
            // Validate with custom validator if provided
            if (pattern.validator && !pattern.validator(candidate_nodes)) {
                continue;
            }

            // Match found! Fuse the nodes (marks nodes for deletion)
            auto status = fuse_nodes(graph, candidate_nodes, pattern, nodes_to_delete);
            if (status == core::Status::SUCCESS) {
                num_fused++;

                MI_LOG_INFO("[FusionPass] Fused pattern '" + pattern.name + "': " +
                            candidate_nodes[0]->name() + " -> " + candidate_nodes.back()->name());
            }
        }
    }

    // Phase 2: Sweep - Batch deletion
    for (const auto& node_name : nodes_to_delete) {
        graph->remove_node(node_name);
    }

    return core::Status::SUCCESS;
}

bool FusionPass::match_pattern(const std::vector<std::shared_ptr<Node>>& nodes,
                               const FusionPattern& pattern) {
    if (nodes.size() != pattern.operator_sequence.size()) {
        return false;
    }

    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!nodes[i] || !nodes[i]->get_operator()) {
            return false;
        }
        if (nodes[i]->get_operator()->name() != pattern.operator_sequence[i]) {
            return false;
        }
    }

    return true;
}

core::Status FusionPass::fuse_nodes(Graph* graph, const std::vector<std::shared_ptr<Node>>& nodes,
                                    const FusionPattern& pattern,
                                    std::unordered_set<std::string>& nodes_to_delete) {
    if (nodes.empty() || !graph) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    if (pattern.fused_operator_type == "ConvActivation") {
        /**
         * TensorRT-style Fusion: Set activation directly on Conv2D
         *
         * TensorRT approach:
         * - Does NOT create a separate fused layer (e.g., "ConvReLU")
         * - Instead, sets activation on the Conv layer: conv->setActivation(type)
         * - The Conv layer internally fuses the activation
         * - Supports all activation types: ReLU, Sigmoid, Tanh, LeakyReLU, etc.
         *
         * Our approach (100% aligned with TensorRT):
         * 1. Get the Conv2D node and activation node
         * 2. Determine activation type from operator name
         * 3. Set activation on Conv2D: conv->set_activation(type)
         * 4. Remove the activation node from graph
         * 5. Reconnect Conv2D directly to activation's consumers
         */

        auto conv_node = nodes[0];
        auto activation_node = nodes[1];

        auto conv_op = std::dynamic_pointer_cast<operators::Conv2D>(conv_node->get_operator());
        auto activation_op = activation_node->get_operator();

        if (!conv_op || !activation_op) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // TensorRT-style: Map activation operator name to ActivationType
        const std::string& act_name = activation_op->name();
        operators::ActivationType act_type = map_activation_name_to_type(act_name);

        if (act_type == operators::ActivationType::NONE) {
            MI_LOG_WARNING("[FusionPass] Unknown activation type: " + act_name +
                           ", skipping fusion");
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Set activation on Conv2D (TensorRT-style)
        conv_op->set_activation(act_type);

        MI_LOG_INFO("[FusionPass] Fused activation into Conv2D: " + conv_node->name() +
                    ".set_activation(" + act_name + ")");

        // Reconnect graph: Conv2D → Activation's consumers
        const auto& activation_outputs = activation_node->outputs();
        for (const auto& output_node : activation_outputs) {
            if (!output_node) {
                continue;
            }
            auto& consumer_inputs = output_node->mutable_inputs();
            consumer_inputs.erase(std::remove_if(consumer_inputs.begin(), consumer_inputs.end(),
                                                 [&](const std::shared_ptr<Node>& n) {
                                                     return n &&
                                                            n->name() == activation_node->name();
                                                 }),
                                  consumer_inputs.end());
            (void)graph->connect(conv_node->name(), output_node->name());
        }

        // Mark activation node for deletion (Deferred Deletion)
        nodes_to_delete.insert(activation_node->name());

    } else {
        MI_LOG_WARNING("[FusionPass] Unknown fused operator type: " + pattern.fused_operator_type);
        return core::Status::ERROR_NOT_IMPLEMENTED;
    }

    return core::Status::SUCCESS;
}

bool FusionPass::try_fuse_conv_activation(
    Graph* graph, std::shared_ptr<Node> conv_node,
    std::unordered_set<std::string>& nodes_to_delete) {  // 参数改名：更清晰的语义

    /**
     * TensorRT-style Conv + Activation fusion
     *
     * Fast path checks:
     * 1. Conv has exactly one consumer
     * 2. Consumer is a supported activation
     * 3. Activation not already marked for deletion
     *
     * If all checks pass: fuse!
     */

    // Fast check: Conv must have exactly one consumer
    if (conv_node->outputs().size() != 1) {
        return false;
    }

    auto activation_node = conv_node->outputs()[0];
    if (!activation_node || !activation_node->get_operator()) {
        return false;
    }

    // Fast check: Already marked for deletion?
    if (nodes_to_delete.count(activation_node->name()) > 0) {
        return false;
    }

    // Fast check: Is it a supported activation?
    const std::string& act_name = activation_node->get_operator()->name();
    operators::ActivationType act_type = map_activation_name_to_type(act_name);
    if (act_type == operators::ActivationType::NONE) {
        return false;  // Not a supported activation
    }

    // All checks passed! Perform fusion
    auto conv_op = std::dynamic_pointer_cast<operators::Conv2D>(conv_node->get_operator());
    if (!conv_op) {
        return false;
    }

    // TensorRT-style: Set activation on Conv2D
    conv_op->set_activation(act_type);

    MI_LOG_INFO("[FusionPass] Fused activation into Conv2D: " + conv_node->name() +
                ".set_activation(" + act_name + ")");

    // ========================================================================
    // Reconnect graph: Conv2D → Activation's consumers
    // ========================================================================

    // Step 1: Remove activation from conv's outputs
    auto& conv_outputs = conv_node->mutable_outputs();
    conv_outputs.erase(std::remove_if(conv_outputs.begin(), conv_outputs.end(),
                                      [&](const std::shared_ptr<Node>& n) {
                                          return n && n->name() == activation_node->name();
                                      }),
                       conv_outputs.end());

    // Step 2: For each consumer, replace activation with conv
    const auto& activation_outputs = activation_node->outputs();
    for (const auto& output_node : activation_outputs) {
        if (!output_node) {
            continue;
        }

        // Remove activation from consumer's inputs
        auto& consumer_inputs = output_node->mutable_inputs();
        consumer_inputs.erase(std::remove_if(consumer_inputs.begin(), consumer_inputs.end(),
                                             [&](const std::shared_ptr<Node>& n) {
                                                 return n && n->name() == activation_node->name();
                                             }),
                              consumer_inputs.end());

        // Connect conv to consumer
        (void)graph->connect(conv_node->name(), output_node->name());
    }

    // ========================================================================
    // Mark activation node for deletion (Deferred Deletion)
    // ========================================================================
    // IMPORTANT: We don't call graph->remove_node() here!
    // Instead, we mark it for deletion in Phase 2 (in apply function)
    nodes_to_delete.insert(activation_node->name());

    return true;
}

bool FusionPass::validate_conv_activation(const std::vector<std::shared_ptr<Node>>& nodes) {
    if (nodes.size() != 2) {
        return false;
    }

    auto conv_node = nodes[0];
    auto activation_node = nodes[1];

    // Verify Conv2D operator
    auto conv_op = std::dynamic_pointer_cast<operators::Conv2D>(conv_node->get_operator());
    if (!conv_op) {
        return false;
    }

    // Verify activation operator exists
    auto activation_op = activation_node->get_operator();
    if (!activation_op) {
        return false;
    }

    // Check if it's a supported activation type
    const std::string& act_name = activation_op->name();
    operators::ActivationType act_type = map_activation_name_to_type(act_name);

    if (act_type == operators::ActivationType::NONE) {
        return false;  // Unknown or unsupported activation
    }

    // Check that Conv has exactly one consumer (the activation)
    if (conv_node->outputs().size() != 1) {
        return false;
    }

    // Check that activation consumes Conv's output
    if (conv_node->outputs()[0] != activation_node) {
        return false;
    }

    return true;
}

}  // namespace graph
}  // namespace mini_infer
