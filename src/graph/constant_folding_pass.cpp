#include "mini_infer/graph/constant_folding_pass.h"

#include <algorithm>
#include <unordered_set>

#include "mini_infer/core/op_type.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/operators/generic_operator.h"
#include "mini_infer/utils/logger.h"

namespace mini_infer {
namespace graph {

ConstantFoldingPass::ConstantFoldingPass() : OptimizationPass("ConstantFolding") {}

core::Status ConstantFoldingPass::apply(Graph* graph, int& num_modifications) {
    if (!graph) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    num_modifications = 0;

    MI_LOG_INFO("[ConstantFolding] Starting constant folding pass...");

    // Step 1: Collect all initializers into Graph::constants_
    collect_initializers(graph);

    // Step 2: Topological traversal to fold constant subgraphs
    std::vector<std::shared_ptr<Node>> sorted_nodes;
    auto status = graph->topological_sort(sorted_nodes);
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("[ConstantFolding] Failed to perform topological sort");
        return status;
    }

    std::unordered_set<size_t> nodes_to_delete;

    for (const auto& node : sorted_nodes) {
        if (!node || !node->get_operator()) {
            continue;
        }

        // Skip if already marked for deletion
        if (nodes_to_delete.count(node->id()) > 0) {
            continue;
        }

        // Skip graph inputs (they are runtime inputs, not constants)
        if (graph->is_input(node->name())) {
            continue;
        }

        // Check if this node can be folded
        if (!can_fold_node(node, graph)) {
            continue;
        }

        // Execute node on CPU to get constant output
        std::vector<std::shared_ptr<core::Tensor>> output_tensors;
        status = execute_node_on_cpu(node, graph, output_tensors);
        if (status != core::Status::SUCCESS) {
            MI_LOG_WARNING("[ConstantFolding] Failed to fold node: " + node->name());
            continue;
        }

        // Register outputs as constants
        if (!output_tensors.empty() && output_tensors[0]) {
            graph->set_constant(node->id(), output_tensors[0]);

            MI_LOG_INFO("[ConstantFolding] Folded node: " + node->name() +
                        " (type: " + std::string(node->type_name()) +
                        ", output shape: " + output_tensors[0]->shape().to_string() + ")");

            // Mark node for deletion
            nodes_to_delete.insert(node->id());
            num_modifications++;
        }
    }

    // Step 3: Update graph outputs if any deleted node was an output
    auto outputs = graph->outputs();
    for (auto& output_name : outputs) {
        auto output_node = graph->get_node(output_name);
        if (output_node && nodes_to_delete.count(output_node->id()) > 0) {
            // If output node is folded, keep it (don't delete)
            // The constant value will be used at runtime
            nodes_to_delete.erase(output_node->id());
            MI_LOG_INFO("[ConstantFolding] Keeping folded output node: " + output_name);
        }
    }

    // Step 4: Remove folded nodes (except outputs)
    for (const auto& node_id : nodes_to_delete) {
        auto node = graph->get_node(node_id);
        if (node) {
            graph->remove_node(node->name());
        }
    }

    if (num_modifications > 0) {
        MI_LOG_INFO("[ConstantFolding] Folded " + std::to_string(num_modifications) +
                    " node(s), removed " + std::to_string(nodes_to_delete.size()) + " node(s)");
    } else {
        MI_LOG_INFO("[ConstantFolding] No constant nodes to fold");
    }

    return core::Status::SUCCESS;
}

bool ConstantFoldingPass::can_fold_node(const std::shared_ptr<Node>& node, Graph* graph) const {
    if (!node || !graph) {
        return false;
    }

    // Check if all inputs are constants
    const auto& input_edges = node->inputs();
    const auto& input_tensors = node->input_tensors();

    // Case 1: Node has no graph edges (only imported tensors)
    if (input_edges.empty()) {
        // All imported tensors must be constants
        for (const auto& tensor : input_tensors) {
            if (!tensor) {
                return false;
            }
        }
        return !input_tensors.empty();
    }

    // Case 2: Node has graph edges - check if all predecessor outputs are constants
    for (const auto& edge : input_edges) {
        if (!edge.node) {
            continue;
        }

        // Check if predecessor's output is a constant (using node ID)
        if (!graph->is_constant(edge.node->id())) {
            return false;
        }
    }

    // Also check imported tensors (weights/biases)
    // These are typically already constants, but we verify
    for (const auto& tensor : input_tensors) {
        if (tensor && tensor->empty()) {
            return false;
        }
    }

    return true;
}

core::Status ConstantFoldingPass::execute_node_on_cpu(
    const std::shared_ptr<Node>& node, Graph* graph,
    std::vector<std::shared_ptr<core::Tensor>>& output_tensors) {

    if (!node || !graph) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Get CPU plugin for this operator
    auto plugin = operators::PluginRegistry::instance().create_plugin(
        node->type(), core::DeviceType::CPU);

    if (!plugin) {
        MI_LOG_WARNING("[ConstantFolding] No CPU plugin for operator: " +
                       std::string(node->type_name()));
        return core::Status::ERROR_NOT_IMPLEMENTED;
    }

    // Transfer parameters from GenericOperator to plugin
    auto* generic_op = dynamic_cast<operators::GenericOperator*>(node->get_operator().get());
    if (generic_op && generic_op->plugin_param()) {
        plugin->set_param(generic_op->plugin_param());
    }

    // Get input tensors
    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    if (!get_node_inputs(node, graph, input_tensors)) {
        return core::Status::ERROR_RUNTIME;
    }

    // Prepare input shapes and dtypes for shape inference
    std::vector<core::Shape> input_shapes;
    std::vector<core::DataType> input_dtypes;
    for (const auto& tensor : input_tensors) {
        if (tensor) {
            input_shapes.push_back(tensor->shape());
            input_dtypes.push_back(tensor->dtype());
        }
    }

    // Infer output shapes
    std::vector<core::Shape> output_shapes;
    std::vector<core::DataType> output_dtypes;
    auto status = plugin->infer_output_shapes_with_tensors(
        input_shapes, input_dtypes, input_tensors, output_shapes, output_dtypes);

    if (status != core::Status::SUCCESS) {
        MI_LOG_WARNING("[ConstantFolding] Shape inference failed for node: " + node->name());
        return status;
    }

    // Allocate output tensors (CPU)
    output_tensors.clear();
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        auto dtype = (i < output_dtypes.size()) ? output_dtypes[i] : core::DataType::FLOAT32;
        auto output_tensor = std::make_shared<core::Tensor>(
            output_shapes[i], dtype, core::DeviceType::CPU);
        output_tensors.push_back(output_tensor);
    }

    // Execute plugin on CPU
    operators::PluginContext context;
    context.device_context = nullptr;  // CPU execution doesn't need device context
    status = plugin->enqueue(input_tensors, output_tensors, context);
    if (status != core::Status::SUCCESS) {
        MI_LOG_WARNING("[ConstantFolding] Execution failed for node: " + node->name());
        return status;
    }

    return core::Status::SUCCESS;
}

void ConstantFoldingPass::collect_initializers(Graph* graph) {
    if (!graph) {
        return;
    }

    const auto& all_nodes = graph->nodes();
    int initializer_count = 0;

    for (const auto& node : all_nodes) {
        if (!node) {
            continue;
        }

        // Collect initializers (nodes with input_tensors but no graph inputs)
        const auto& input_tensors = node->input_tensors();
        const auto& input_edges = node->inputs();

        // If node has imported tensors (weights/biases) and no graph inputs,
        // it's a constant node - register the node itself as constant
        if (!input_tensors.empty() && input_edges.empty()) {
            // For nodes with only imported tensors, register the node as constant
            // This allows downstream nodes to be folded
            if (!input_tensors.empty() && input_tensors[0]) {
                graph->set_constant(node->id(), input_tensors[0]);
                initializer_count++;
            }
        }
    }

    if (initializer_count > 0) {
        MI_LOG_INFO("[ConstantFolding] Collected " + std::to_string(initializer_count) +
                    " initializer(s)");
    }
}

bool ConstantFoldingPass::get_node_inputs(
    const std::shared_ptr<Node>& node, Graph* graph,
    std::vector<std::shared_ptr<core::Tensor>>& input_tensors) const {

    if (!node || !graph) {
        return false;
    }

    input_tensors.clear();

    const auto& input_edges = node->inputs();
    const auto& imported_tensors = node->input_tensors();

    // Case 1: Node has graph edges
    if (!input_edges.empty()) {
        // Find max dst_port to determine input count
        int max_dst_port = -1;
        for (const auto& edge : input_edges) {
            max_dst_port = std::max(max_dst_port, edge.dst_port);
        }

        if (max_dst_port >= 0) {
            input_tensors.resize(max_dst_port + 1);
        }

        // Fill from graph edges (predecessor outputs)
        for (const auto& edge : input_edges) {
            if (!edge.node || edge.dst_port < 0) {
                continue;
            }

            // Get constant tensor from predecessor (using node ID)
            auto constant_tensor = graph->get_constant(edge.node->id());
            if (!constant_tensor) {
                MI_LOG_WARNING("[ConstantFolding] Missing constant for node: " +
                               edge.node->name());
                return false;
            }

            if (static_cast<size_t>(edge.dst_port) < input_tensors.size()) {
                input_tensors[edge.dst_port] = constant_tensor;
            }
        }

        // Fill remaining slots from imported tensors
        for (size_t i = 0; i < imported_tensors.size() && i < input_tensors.size(); ++i) {
            if (!input_tensors[i] && imported_tensors[i]) {
                input_tensors[i] = imported_tensors[i];
            }
        }

        // Append any extra imported tensors
        for (size_t i = input_tensors.size(); i < imported_tensors.size(); ++i) {
            if (imported_tensors[i]) {
                input_tensors.push_back(imported_tensors[i]);
            }
        }
    } else {
        // Case 2: No graph edges, use imported tensors only
        input_tensors = imported_tensors;
    }

    // Verify all inputs are available
    for (const auto& tensor : input_tensors) {
        if (!tensor || tensor->empty()) {
            return false;
        }
    }

    return !input_tensors.empty();
}

}  // namespace graph
}  // namespace mini_infer

// Auto-register ConstantFoldingPass with priority 50 (before fusion)
namespace {
std::shared_ptr<mini_infer::graph::OptimizationPass> create_ConstantFoldingPass() {
    return std::make_shared<mini_infer::graph::ConstantFoldingPass>();
}
struct ConstantFoldingPass_Register {
    ConstantFoldingPass_Register() {
        mini_infer::graph::OptimizationPassRegistry::instance().register_pass(
            "ConstantFolding", create_ConstantFoldingPass, 50);
    }
};
static ConstantFoldingPass_Register g_ConstantFoldingPass_register;
}  // namespace
