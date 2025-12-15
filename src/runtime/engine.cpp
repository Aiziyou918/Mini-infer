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
    if (config_.enable_dynamic_shapes && config_.optimization_profile) {
        // Use optimal shapes from profile for build-time inference
        status = infer_shapes_with_profile();
    } else {
        // Traditional static shape inference
        status = infer_shapes();
    }
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
    
    // Step 6: Initialize shape inference engine for runtime
    if (config_.enable_dynamic_shapes) {
        MI_LOG_INFO("[Engine] Step 6: Initializing runtime shape inference...");
        shape_inference_engine_ = std::make_unique<ShapeInferenceEngine>(graph_);
        shape_inference_engine_->set_verbose(config_.enable_profiling);
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
    // TensorRT-style: Iterate through nodes in topological order and infer output shapes
    int total_inferred = 0;
    int failed_count = 0;
    
    for (auto& node : sorted_nodes_) {
        if (!node) {
            continue;
        }
        
        // Skip nodes without operator (e.g., input placeholders)
        if (!node->get_operator()) {
            continue;
        }
        
        // Collect input shapes from graph connections (data tensors)
        std::vector<core::Shape> input_shapes;
        for (const auto& input_node : node->inputs()) {
            if (input_node && !input_node->output_tensors().empty()) {
                const auto& out_tensor = input_node->output_tensors()[0];
                if (out_tensor) {
                    input_shapes.push_back(out_tensor->shape());
                }
            }
        }
        
        // Merge with imported input tensors (weights/bias)
        // For operators like Conv2D: [data, weight, bias]
        // Graph connections provide data, imported tensors provide weight/bias
        const auto& imported_inputs = node->input_tensors();
        if (!imported_inputs.empty()) {
            // If we have graph inputs, they override the first entries
            size_t graph_input_count = input_shapes.size();
            
            // Add shapes from imported tensors (weights, bias, etc.)
            for (size_t i = graph_input_count; i < imported_inputs.size(); ++i) {
                if (imported_inputs[i]) {
                    input_shapes.push_back(imported_inputs[i]->shape());
                }
            }
            
            // If no graph inputs but have imported inputs, use all imported
            if (graph_input_count == 0) {
                input_shapes.clear();
                for (const auto& tensor : imported_inputs) {
                    if (tensor) {
                        input_shapes.push_back(tensor->shape());
                    }
                }
            }
        }
        
        // Check if we have enough input shapes
        if (input_shapes.empty()) {
            if (config_.enable_profiling) {
                MI_LOG_WARNING("[Engine] Node " + node->name() + 
                              " has no input shapes, skipping shape inference");
            }
            failed_count++;
            continue;
        }
        
        // Infer output shapes
        std::vector<core::Shape> output_shapes;
        auto status = node->get_operator()->infer_shape(input_shapes, output_shapes);
        
        if (status != core::Status::SUCCESS) {
            MI_LOG_WARNING("[Engine] Failed to infer shape for node: " + node->name());
            failed_count++;
            continue;
        }
        
        // Validate output shapes
        if (output_shapes.empty()) {
            MI_LOG_WARNING("[Engine] Node " + node->name() + 
                          " produced empty output shapes");
            failed_count++;
            continue;
        }
        
        // Create output tensors if needed and set their shapes
        auto& output_tensors = node->output_tensors();
        
        // Ensure we have enough output tensor slots
        while (output_tensors.size() < output_shapes.size()) {
            output_tensors.push_back(std::make_shared<core::Tensor>());
        }
        
        // Update output tensors with inferred shapes
        for (size_t i = 0; i < output_shapes.size() && i < output_tensors.size(); ++i) {
            // Create new tensor with inferred shape (don't allocate data yet)
            // Note: reshape() doesn't work on empty tensors, so we create new ones
            auto new_tensor = std::make_shared<core::Tensor>();
            
            // Manually set shape by creating a dummy tensor then extracting shape
            // This is a workaround since Tensor doesn't have a direct set_shape method
            if (output_shapes[i].numel() > 0) {
                // For now, just create a properly shaped tensor without allocating
                // We'll allocate in allocate_tensors() step
                auto temp = std::make_shared<core::Tensor>(
                    output_shapes[i], 
                    core::DataType::FLOAT32
                );
                output_tensors[i] = temp;
            } else {
                output_tensors[i] = new_tensor;
            }
            
            total_inferred++;
            
            if (config_.enable_profiling) {
                MI_LOG_INFO("[Engine] Node " + node->name() + " output[" + 
                           std::to_string(i) + "] shape: " + output_shapes[i].to_string());
            }
        }
    }
    
    MI_LOG_INFO("[Engine] Shape inference completed: " + 
                std::to_string(total_inferred) + " tensor(s) inferred, " +
                std::to_string(failed_count) + " failed");
    
    return core::Status::SUCCESS;
}

core::Status Engine::infer_shapes_with_profile() {
    if (!config_.optimization_profile) {
        MI_LOG_ERROR("[Engine] Optimization profile is null");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Get optimal shapes from profile
    auto optimal_shapes = config_.optimization_profile->get_optimal_shapes();
    
    if (config_.enable_profiling) {
        MI_LOG_INFO("[Engine] Using optimization profile with optimal shapes:");
        for (const auto& [name, shape] : optimal_shapes) {
            MI_LOG_INFO("[Engine]   " + name + ": " + shape.to_string());
        }
    }
    
    // Set optimal shapes for graph inputs
    for (const auto& input_name : graph_->inputs()) {
        auto it = optimal_shapes.find(input_name);
        if (it == optimal_shapes.end()) {
            MI_LOG_WARNING("[Engine] No optimal shape for input '" + input_name + "'");
            continue;
        }
        
        auto node = graph_->get_node(input_name);
        if (node) {
            // Create or update input tensor with optimal shape
            if (node->output_tensors().empty() || !node->output_tensors()[0]) {
                auto tensor = std::make_shared<core::Tensor>(
                    it->second,
                    core::DataType::FLOAT32
                );
                node->set_output_tensors({tensor});
            } else {
                // Update existing tensor's shape
                auto tensor = std::make_shared<core::Tensor>(
                    it->second,
                    node->output_tensors()[0]->dtype()
                );
                node->set_output_tensors({tensor});
            }
            
            if (config_.enable_profiling) {
                MI_LOG_INFO("[Engine] Set input '" + input_name + 
                           "' shape: " + it->second.to_string());
            }
        }
    }
    
    // Now perform normal shape inference (will use the optimal input shapes)
    return infer_shapes();
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
    
    // Check for dynamic shape changes
    if (config_.enable_dynamic_shapes && shape_inference_engine_) {
        if (check_shape_change(inputs)) {
            MI_LOG_INFO("[Engine] Input shape changed, re-inferring shapes...");
            auto status = handle_shape_change(inputs);
            if (status != core::Status::SUCCESS) {
                MI_LOG_ERROR("[Engine] Failed to handle shape change");
                return status;
            }
        }
    }
    
    // Set input tensors and validate shapes
    for (const auto& input_name : graph_->inputs()) {
        auto it = inputs.find(input_name);
        if (it == inputs.end()) {
            MI_LOG_ERROR("[Engine] Missing input: " + input_name);
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
        
        auto node = graph_->get_node(input_name);
        if (node) {
            // Validate input shape if expected shape is known
            if (!node->output_tensors().empty() && node->output_tensors()[0]) {
                const auto& expected_shape = node->output_tensors()[0]->shape();
                const auto& actual_shape = it->second->shape();
                
                // Check shape compatibility (allow dynamic batch size)
                if (expected_shape.ndim() > 0 && expected_shape.ndim() == actual_shape.ndim()) {
                    bool compatible = true;
                    for (size_t i = 0; i < expected_shape.ndim(); ++i) {
                        // Skip dynamic dimensions (-1) or batch dimension (index 0)
                        if (expected_shape[i] < 0 || i == 0) continue;
                        
                        if (expected_shape[i] != actual_shape[i]) {
                            MI_LOG_ERROR("[Engine] Input '" + input_name + 
                                        "' shape mismatch: expected " + expected_shape.to_string() +
                                        ", got " + actual_shape.to_string());
                            compatible = false;
                            break;
                        }
                    }
                    
                    if (!compatible) {
                        return core::Status::ERROR_INVALID_ARGUMENT;
                    }
                }
            }
            
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
    int skipped_count = 0;
    int failed_count = 0;
    
    for (auto& node : sorted_nodes_) {
        if (!node) continue;
        
        auto& output_tensors = node->output_tensors();
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto& tensor = output_tensors[i];
            if (!tensor) {
                if (config_.enable_profiling) {
                    MI_LOG_WARNING("[Engine] Node " + node->name() + " output[" + 
                                  std::to_string(i) + "] is null");
                }
                failed_count++;
                continue;
            }
            
            // Skip if already allocated (e.g., weights, shared tensors)
            if (!tensor->empty()) {
                skipped_count++;
                continue;
            }
            
            // Validate shape before allocation
            const auto& shape = tensor->shape();
            if (shape.ndim() == 0) {
                if (config_.enable_profiling) {
                    MI_LOG_WARNING("[Engine] Node " + node->name() + " output[" + 
                                  std::to_string(i) + "] has empty shape, skipping allocation");
                }
                failed_count++;
                continue;
            }
            
            if (shape.numel() <= 0) {
                MI_LOG_ERROR("[Engine] Node " + node->name() + " output[" + 
                            std::to_string(i) + "] has invalid shape: " + shape.to_string());
                failed_count++;
                continue;
            }
            
            // Allocate based on shape
            // For now, allocate independently
            // TODO: Use memory pools from memory_plan_
            try {
                *tensor = core::Tensor(shape, tensor->dtype());
                allocated_count++;
                
                if (config_.enable_profiling) {
                    MI_LOG_INFO("[Engine] Allocated tensor for " + node->name() + 
                               " output[" + std::to_string(i) + "]: " + shape.to_string() +
                               " (" + std::to_string(tensor->size_in_bytes() / 1024.0) + " KB)");
                }
            } catch (const std::exception& e) {
                MI_LOG_ERROR("[Engine] Failed to allocate tensor for " + node->name() + 
                            " output[" + std::to_string(i) + "]: " + e.what());
                failed_count++;
            }
        }
    }
    
    MI_LOG_INFO("[Engine] Tensor allocation completed: " + 
                std::to_string(allocated_count) + " allocated, " +
                std::to_string(skipped_count) + " skipped, " +
                std::to_string(failed_count) + " failed");
    
    if (failed_count > 0) {
        MI_LOG_WARNING("[Engine] Some tensors failed to allocate, inference may fail");
    }
    
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

bool Engine::check_shape_change(
    const std::unordered_map<std::string, std::shared_ptr<core::Tensor>>& inputs
) {
    // Extract current input shapes
    std::unordered_map<std::string, core::Shape> current_shapes;
    for (const auto& [name, tensor] : inputs) {
        current_shapes[name] = tensor->shape();
    }
    
    // Check if shapes changed
    return shape_inference_engine_->shapes_changed(current_shapes);
}

core::Status Engine::handle_shape_change(
    const std::unordered_map<std::string, std::shared_ptr<core::Tensor>>& inputs
) {
    // Validate shapes against optimization profile
    if (config_.optimization_profile) {
        std::map<std::string, core::Shape> input_shapes_map;
        for (const auto& [name, tensor] : inputs) {
            input_shapes_map[name] = tensor->shape();
        }
        
        if (!config_.optimization_profile->is_valid_for(input_shapes_map)) {
            MI_LOG_ERROR("[Engine] Input shapes are outside optimization profile range");
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }
    
    // Re-infer shapes
    std::unordered_map<std::string, core::Shape> input_shapes;
    for (const auto& [name, tensor] : inputs) {
        input_shapes[name] = tensor->shape();
    }
    
    auto status = shape_inference_engine_->infer_shapes(input_shapes);
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("[Engine] Runtime shape inference failed");
        return status;
    }
    
    // Get tensors that need reallocation
    auto tensors_to_reallocate = shape_inference_engine_->get_tensors_needing_reallocation();
    
    if (!tensors_to_reallocate.empty()) {
        MI_LOG_INFO("[Engine] Reallocating " + std::to_string(tensors_to_reallocate.size()) + 
                   " tensor(s) due to shape change");
        
        // Reallocate tensors with new shapes
        for (const auto& tensor_name : tensors_to_reallocate) {
            auto node = graph_->get_node(tensor_name);
            if (!node) continue;
            
            auto new_shape = shape_inference_engine_->get_inferred_shape(tensor_name);
            if (!new_shape) continue;
            
            // Create new tensor with inferred shape
            if (!node->output_tensors().empty() && node->output_tensors()[0]) {
                auto dtype = node->output_tensors()[0]->dtype();
                auto new_tensor = std::make_shared<core::Tensor>(*new_shape, dtype);
                node->set_output_tensors({new_tensor});
                
                if (config_.enable_profiling) {
                    MI_LOG_INFO("[Engine]   Reallocated '" + tensor_name + 
                               "': " + new_shape->to_string());
                }
            }
        }
    }
    
    return core::Status::SUCCESS;
}

} // namespace runtime
} // namespace mini_infer

