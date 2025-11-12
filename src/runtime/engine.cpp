#include "mini_infer/runtime/engine.h"
#include "mini_infer/utils/logger.h"
#include <sstream>

namespace mini_infer {
namespace runtime {

Engine::Engine(const EngineConfig& config) 
    : config_(config) {
    backend_ = backends::BackendFactory::create_backend(config_.device_type);
}

core::Status Engine::build(std::shared_ptr<graph::Graph> graph) {
    if (!graph) {
        MI_LOG_ERROR("Graph is null");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    graph_ = graph;
    
    // Graph optimization (assumes graph structure already consistent)
    auto status = graph_->optimize();
    if (status != core::Status::SUCCESS) {
        MI_LOG_WARNING("Graph optimization failed, using original graph");
    }
    
    // Topological sort with validation
    status = graph_->checked_topological_sort(sorted_nodes_);
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("Topological sort failed");
        return status;
    }
    
    // Infer all nodes output shape
    status = allocate_tensors();
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("Tensor allocation failed");
        return status;
    }
    
    MI_LOG_INFO("Engine built successfully");
    return core::Status::SUCCESS;
}

core::Status Engine::forward(
    const std::unordered_map<std::string, std::shared_ptr<core::Tensor>>& inputs,
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>>& outputs) {
    
    if (!graph_) {
        return core::Status::ERROR_RUNTIME;
    }
    
    // Set input tensors
    for (const auto& input_name : graph_->inputs()) {
        auto it = inputs.find(input_name);
        if (it == inputs.end()) {
            MI_LOG_ERROR("Missing input: " + input_name);
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
        
        auto node = graph_->get_node(input_name);
        if (node) {
            node->set_output_tensors({it->second});
        }
    }
    
    // Execute all nodes
    for (auto& node : sorted_nodes_) {
        auto status = execute_node(node);
        if (status != core::Status::SUCCESS) {
            MI_LOG_ERROR("Node execution failed: " + node->name());
            return status;
        }
    }
    
    // Collect output tensors
    outputs.clear();
    for (const auto& output_name : graph_->outputs()) {
        auto node = graph_->get_node(output_name);
        if (node && !node->output_tensors().empty()) {
            outputs[output_name] = node->output_tensors()[0];
        }
    }
    
    return core::Status::SUCCESS;
}

std::vector<std::string> Engine::get_input_names() const {
    if (graph_) {
        return graph_->inputs();
    }
    return {};
}

std::vector<std::string> Engine::get_output_names() const {
    if (graph_) {
        return graph_->outputs();
    }
    return {};
}

std::string Engine::get_profiling_info() const {
    // TODO: Implement profiling info collection
    return "Profiling not implemented yet";
}

core::Status Engine::allocate_tensors() {
    // TODO: Allocate tensors for each node
    // Depends on the operator's infer_shape
    return core::Status::SUCCESS;
}

core::Status Engine::execute_node(std::shared_ptr<graph::Node> node) {
    if (!node || !node->get_operator()) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Collect input tensors
    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (const auto& input_node : node->inputs()) {
        const auto& outputs = input_node->output_tensors();
        if (!outputs.empty()) {
            input_tensors.push_back(outputs[0]);
        }
    }
    
    // Execute operator
    auto& output_tensors = node->output_tensors();
    return node->get_operator()->forward(input_tensors, output_tensors);
}

} // namespace runtime
} // namespace mini_infer

