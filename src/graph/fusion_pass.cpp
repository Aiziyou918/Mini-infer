#include "mini_infer/graph/fusion_pass.h"

#include <algorithm>
#include <unordered_set>

#include "mini_infer/core/op_type.h"
#include "mini_infer/operators/activation_type.h"
#include "mini_infer/operators/conv2d.h"
#include "mini_infer/utils/logger.h"

namespace mini_infer {
namespace graph {

FusionPass::FusionPass() : OptimizationPass("FusionPass") {}

core::Status FusionPass::apply(Graph* graph, int& num_modifications) {
    if (!graph) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    num_modifications = 0;

    // TensorRT-style: Two-Phase Deferred Deletion
    // Phase 1: Traverse and fuse (mark nodes for deletion)
    // Phase 2: Batch delete marked nodes

    const auto& all_nodes = graph->nodes();
    std::unordered_set<std::string> nodes_to_delete;

    // Phase 1: Mark - Traverse and fuse
    for (const auto& [name, node] : all_nodes) {
        if (!node || !node->get_operator()) {
            continue;
        }

        if (nodes_to_delete.count(name) > 0) {
            continue;
        }

        // TensorRT-style: Direct OpType comparison
        if (node->type() == core::OpType::kCONVOLUTION) {
            if (try_fuse_conv_activation(graph, node, nodes_to_delete)) {
                num_modifications++;
            }
            // Future: try_fuse_conv_bn(), try_fuse_conv_bn_activation()
        }
        // Future: kGEMM, kMATMUL fusions
    }

    // Phase 1.5: Update graph outputs if any deleted node was an output
    auto outputs = graph->outputs();
    bool outputs_changed = false;
    for (auto& output_name : outputs) {
        if (nodes_to_delete.count(output_name) > 0) {
            // Find the node that will be deleted and get its predecessor
            auto node = graph->get_node(output_name);
            if (node && !node->inputs().empty()) {
                auto& input_edge = node->inputs()[0];
                if (input_edge.node) {
                    output_name = input_edge.node->name();
                    outputs_changed = true;
                }
            }
        }
    }
    if (outputs_changed) {
        graph->set_outputs(outputs);
    }

    // Phase 2: Sweep - Batch deletion
    for (const auto& node_name : nodes_to_delete) {
        graph->remove_node(node_name);
    }

    if (num_modifications > 0) {
        MI_LOG_INFO("[FusionPass] Applied " + std::to_string(num_modifications) +
                    " fusion(s), removed " + std::to_string(nodes_to_delete.size()) + " node(s)");
    }

    return core::Status::SUCCESS;
}

bool FusionPass::try_fuse_conv_activation(Graph* graph, std::shared_ptr<Node> conv_node,
                                          std::unordered_set<std::string>& nodes_to_delete) {
    // Check: Conv must have exactly one consumer
    if (conv_node->outputs().size() != 1) {
        return false;
    }

    auto activation_node = conv_node->outputs()[0].node;
    if (!activation_node || !activation_node->get_operator()) {
        return false;
    }

    // Check: Not already marked for deletion
    if (nodes_to_delete.count(activation_node->name()) > 0) {
        return false;
    }

    // Check: Consumer is a supported activation (OpType -> ActivationType)
    operators::ActivationType act_type;
    if (!core::op_type_to_activation_type(activation_node->type(), act_type)) {
        return false;
    }

    // Check: Valid Conv2D operator
    auto conv_op = std::dynamic_pointer_cast<operators::Conv2D>(conv_node->get_operator());
    if (!conv_op) {
        return false;
    }

    // TensorRT-style: Set activation on Conv2D
    conv_op->set_activation(act_type);

    MI_LOG_INFO("[FusionPass] Fused: " + conv_node->name() + " + " +
                std::string(activation_node->type_name()));

    // Reconnect graph: Conv -> Activation's consumers
    auto& conv_outputs = conv_node->mutable_outputs();
    conv_outputs.erase(std::remove_if(conv_outputs.begin(), conv_outputs.end(),
                                      [&](const Node::Edge& e) {
                                          return e.node &&
                                                 e.node->name() == activation_node->name();
                                      }),
                       conv_outputs.end());

    for (const auto& edge : activation_node->outputs()) {
        if (!edge.node)
            continue;

        auto& consumer_inputs = edge.node->mutable_inputs();
        consumer_inputs.erase(std::remove_if(consumer_inputs.begin(), consumer_inputs.end(),
                                             [&](const Node::Edge& e) {
                                                 return e.node &&
                                                        e.node->name() == activation_node->name();
                                             }),
                              consumer_inputs.end());

        (void)graph->connect(conv_node->name(), edge.node->name(), edge.src_port, edge.dst_port);
    }

    // Mark for deletion (deferred)
    nodes_to_delete.insert(activation_node->name());

    return true;
}

}  // namespace graph
}  // namespace mini_infer

// Auto-register FusionPass with priority 100 (default)
namespace {
std::shared_ptr<mini_infer::graph::OptimizationPass> create_FusionPass() {
    return std::make_shared<mini_infer::graph::FusionPass>();
}
struct FusionPass_Register {
    FusionPass_Register() {
        mini_infer::graph::OptimizationPassRegistry::instance().register_pass(
            "FusionPass", create_FusionPass, 100);
    }
};
static FusionPass_Register g_FusionPass_register;
}  // namespace
