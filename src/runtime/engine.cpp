#include "mini_infer/runtime/engine.h"
#include "mini_infer/utils/logger.h"

namespace mini_infer {
namespace runtime {

Engine::Engine(const EngineConfig& config) 
    : config_(config) {
    backend_ = backends::BackendFactory::create_backend(config_.device_type);
}

core::Status Engine::build(std::shared_ptr<graph::Graph> graph) {
    if (!graph) {
        MI_LOG_ERROR("[Engine] Graph is null");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    graph_ = graph;
    
    MI_LOG_INFO("[Engine] ========================================");
    MI_LOG_INFO("[Engine] Building Engine");
    MI_LOG_INFO("[Engine] ========================================");
    
    // Step 1: Graph optimization (operator fusion, constant folding, etc.)
    if (config_.enable_graph_optimization) {
        MI_LOG_INFO("[Engine] Step 1: Applying graph optimizations...");
        auto status = optimize_graph();
        if (status != core::Status::SUCCESS) {
            MI_LOG_WARNING("[Engine] Graph optimization failed, using original graph");
        }
    } else {
        MI_LOG_INFO("[Engine] Step 1: Graph optimization disabled");
    }
    
    // Step 2: Topological sort with validation
    MI_LOG_INFO("[Engine] Step 2: Performing topological sort...");
    auto status = graph_->checked_topological_sort(sorted_nodes_);
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("[Engine] Topological sort failed");
        return status;
    }
    MI_LOG_INFO("[Engine] Topological sort completed: " + 
                std::to_string(sorted_nodes_.size()) + " nodes");
    
    // Step 3: Shape inference
    MI_LOG_INFO("[Engine] Step 3: Inferring tensor shapes...");
    status = infer_shapes();
    if (status != core::Status::SUCCESS) {
        MI_LOG_WARNING("[Engine] Shape inference incomplete");
    }
    
    // Step 4: Memory planning (TensorRT-style static memory allocation)
    if (config_.enable_memory_planning) {
        MI_LOG_INFO("[Engine] Step 4: Planning memory allocation...");
        status = plan_memory();
        if (status != core::Status::SUCCESS) {
            MI_LOG_WARNING("[Engine] Memory planning failed, using default allocation");
        }
    } else {
        MI_LOG_INFO("[Engine] Step 4: Memory planning disabled");
    }
    
    // Step 5: Allocate tensors
    MI_LOG_INFO("[Engine] Step 5: Allocating tensors...");
    status = allocate_tensors();
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("[Engine] Tensor allocation failed");
        return status;
    }
    
    MI_LOG_INFO("[Engine] ========================================");
    MI_LOG_INFO("[Engine] Engine built successfully");
    MI_LOG_INFO("[Engine] ========================================");
    return core::Status::SUCCESS;
}

core::Status Engine::optimize_graph() {
    // TensorRT-style: Use registered optimization passes
    // All passes are auto-registered via REGISTER_OPTIMIZATION_PASS macro
    auto optimizer = graph::GraphOptimizer::create_default();
    optimizer.set_verbose(config_.enable_profiling);
    
    auto status = optimizer.optimize(graph_.get());
    optimization_stats_ = optimizer.get_statistics();
    
    if (status == core::Status::SUCCESS) {
        MI_LOG_INFO("[Engine] Graph optimization completed: " + 
                    std::to_string(optimization_stats_.total_modifications) + 
                    " modification(s)");
    }
    
    return status;
}

core::Status Engine::infer_shapes() {
    // Iterate through nodes in topological order and infer output shapes
    for (auto& node : sorted_nodes_) {
        if (!node || !node->get_operator()) {
            continue;
        }
        
        // Collect input shapes
        std::vector<core::Shape> input_shapes;
        for (const auto& input_tensor : node->input_tensors()) {
            if (input_tensor) {
                input_shapes.push_back(input_tensor->shape());
            }
        }
        
        // Infer output shapes
        std::vector<core::Shape> output_shapes;
        auto status = node->get_operator()->infer_shape(input_shapes, output_shapes);
        
        if (status != core::Status::SUCCESS) {
            MI_LOG_WARNING("[Engine] Failed to infer shape for node: " + node->name());
            continue;
        }
        
        // Update output tensors with inferred shapes
        auto& output_tensors = node->output_tensors();
        for (size_t i = 0; i < output_shapes.size() && i < output_tensors.size(); ++i) {
            if (output_tensors[i]) {
                output_tensors[i]->reshape(output_shapes[i]);
            }
        }
    }
    
    return core::Status::SUCCESS;
}

core::Status Engine::plan_memory() {
    MemoryPlanner planner;
    planner.set_enabled(true);
    planner.set_verbose(config_.enable_profiling);
    planner.set_alignment(config_.memory_alignment);
    
    memory_plan_ = planner.plan(graph_.get());
    
    if (memory_plan_.pools.empty()) {
        MI_LOG_WARNING("[Engine] Memory planning produced no pools");
        return core::Status::ERROR_RUNTIME;
    }
    
    MI_LOG_INFO("[Engine] Memory planning completed:");
    MI_LOG_INFO("[Engine]   Original memory:  " + 
                std::to_string(memory_plan_.original_memory / 1024.0) + " KB");
    MI_LOG_INFO("[Engine]   Optimized memory: " + 
                std::to_string(memory_plan_.total_memory / 1024.0) + " KB");
    MI_LOG_INFO("[Engine]   Memory saving:    " + 
                std::to_string(memory_plan_.memory_saving_ratio * 100.0f) + "%");
    MI_LOG_INFO("[Engine]   Number of pools:  " + 
                std::to_string(memory_plan_.pools.size()));
    
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
    
    // Execute all nodes (skip placeholder nodes without operator)
    for (auto& node : sorted_nodes_) {
        if (!node || !node->get_operator()) {
            continue;
        }
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
    // Allocate tensors for nodes based on inferred shapes
    // If memory planning is enabled, tensors will be allocated from memory pools
    // Otherwise, allocate independently
    
    int allocated_count = 0;
    for (auto& node : sorted_nodes_) {
        if (!node) continue;
        
        auto& output_tensors = node->output_tensors();
        for (auto& tensor : output_tensors) {
            if (!tensor) continue;
            
            // Skip if already allocated (e.g., weights)
            if (!tensor->empty()) continue;
            
            // Allocate based on shape
            const auto& shape = tensor->shape();
            if (shape.numel() > 0) {
                // For now, allocate independently
                // TODO: Use memory pools from memory_plan_
                *tensor = core::Tensor(shape, tensor->dtype());
                allocated_count++;
            }
        }
    }
    
    MI_LOG_INFO("[Engine] Allocated " + std::to_string(allocated_count) + " tensors");
    return core::Status::SUCCESS;
}

core::Status Engine::execute_node(std::shared_ptr<graph::Node> node) {
    if (!node) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    // Placeholder nodes (e.g., graph inputs) have no operator; skip execution
    if (!node->get_operator()) {
        return core::Status::SUCCESS;
    }
    
    // Collect input tensors
    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (const auto& input_node : node->inputs()) {
        const auto& outputs = input_node->output_tensors();
        if (!outputs.empty()) {
            input_tensors.push_back(outputs[0]);
        }
    }

    // Merge with importer-captured inputs to include weights/bias, while
    // allowing graph-connected data tensors to override the first entries.
    const auto& imported_inputs = node->input_tensors();
    if (!imported_inputs.empty()) {
        auto merged = imported_inputs; // copy
        if (!input_tensors.empty()) {
            const size_t n = std::min(merged.size(), input_tensors.size());
            for (size_t i = 0; i < n; ++i) {
                merged[i] = input_tensors[i];
            }
        }
        input_tensors.swap(merged);
    }
    
    // Execute operator
    auto& output_tensors = node->output_tensors();
    return node->get_operator()->forward(input_tensors, output_tensors);
}

} // namespace runtime
} // namespace mini_infer

