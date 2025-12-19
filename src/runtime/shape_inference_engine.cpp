#include "mini_infer/runtime/shape_inference_engine.h"

#include <algorithm>

#include "mini_infer/utils/logger.h"

namespace mini_infer {
namespace runtime {

ShapeInferenceEngine::ShapeInferenceEngine(std::shared_ptr<graph::Graph> graph) : graph_(graph) {}

core::Status ShapeInferenceEngine::ensure_sorted() {
    if (sorted_nodes_.empty()) {
        auto status = graph_->checked_topological_sort(sorted_nodes_);
        if (status != core::Status::SUCCESS) {
            MI_LOG_ERROR("[ShapeInferenceEngine] Topological sort failed");
            return status;
        }
    }
    return core::Status::SUCCESS;
}

core::Status ShapeInferenceEngine::infer_shapes(
    const std::unordered_map<std::string, core::Shape>& input_shapes) {
    auto status = ensure_sorted();
    if (status != core::Status::SUCCESS) {
        return status;
    }

    std::vector<RuntimeInputShape> runtime_shapes;
    runtime_shapes.reserve(input_shapes.size());
    for (const auto& [name, shape] : input_shapes) {
        auto node = graph_->get_node(name);
        if (!node) {
            MI_LOG_ERROR("[ShapeInferenceEngine] Input node not found: " + name);
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        RuntimeInputShape binding;
        binding.node_id = node->id();
        binding.shape = shape;
        runtime_shapes.push_back(binding);
    }

    return infer_shapes(runtime_shapes);
}

core::Status ShapeInferenceEngine::infer_shapes(
    const std::vector<RuntimeInputShape>& input_shapes) {
    auto status = ensure_sorted();
    if (status != core::Status::SUCCESS) {
        return status;
    }
    return infer_shapes_internal(input_shapes);
}

core::Status ShapeInferenceEngine::infer_shapes_internal(
    const std::vector<RuntimeInputShape>& input_shapes) {
    if (verbose_) {
        MI_LOG_INFO("[ShapeInferenceEngine] Starting runtime shape inference...");
        MI_LOG_INFO("[ShapeInferenceEngine] Input shapes:");
        for (const auto& binding : input_shapes) {
            if (binding.node_id < sorted_nodes_.size() && sorted_nodes_[binding.node_id]) {
                MI_LOG_INFO("[ShapeInferenceEngine]   " +
                            sorted_nodes_[binding.node_id]->name() + ": " +
                            binding.shape.to_string());
            }
        }
    }

    // Resize storage to fit all nodes
    inferred_shapes_.clear();
    inferred_shapes_.resize(sorted_nodes_.size());

    // Store input shapes (by node ID)
    for (const auto& binding : input_shapes) {
        if (binding.node_id < inferred_shapes_.size()) {
            inferred_shapes_[binding.node_id] = {binding.shape};  // Input nodes have single output
        }
    }

    int total_inferred = 0;

    // Iterate through nodes in topological order
    for (auto& node : sorted_nodes_) {
        auto op = node->get_operator();
        if (!op) {
            continue;  // Skip nodes without operators (e.g., input nodes)
        }

        // Collect input shapes using node IDs (O(1) lookup)
        std::vector<core::Shape> input_shapes_vec;
        bool all_inputs_ready = true;

        // Step 1: Collect shapes from graph connections (data tensors)
        const auto& input_edges = node->inputs();
        size_t graph_input_count = 0;
        if (!input_edges.empty()) {
            int max_dst_port = -1;
            for (const auto& edge : input_edges) {
                max_dst_port = std::max(max_dst_port, edge.dst_port);
            }
            graph_input_count = static_cast<size_t>(max_dst_port + 1);
            input_shapes_vec.resize(graph_input_count);
        }

        for (const auto& edge : input_edges) {
            if (!edge.node || edge.dst_port < 0 || edge.src_port < 0) {
                all_inputs_ready = false;
                break;
            }
            size_t input_id = edge.node->id();
            if (input_id >= inferred_shapes_.size() || inferred_shapes_[input_id].empty()) {
                all_inputs_ready = false;
                break;
            }
            const auto& outputs = inferred_shapes_[input_id];
            const size_t src_index = static_cast<size_t>(edge.src_port);
            const size_t dst_index = static_cast<size_t>(edge.dst_port);
            if (src_index >= outputs.size() || dst_index >= input_shapes_vec.size()) {
                all_inputs_ready = false;
                break;
            }
            input_shapes_vec[dst_index] = outputs[src_index];
        }
        if (all_inputs_ready) {
            for (const auto& shape : input_shapes_vec) {
                if (shape.ndim() == 0) {
                    all_inputs_ready = false;
                    break;
                }
            }
        }

        // Step 2: Append shapes from imported tensors (weights/bias)
        const auto& imported_tensors = node->input_tensors();
        for (size_t i = graph_input_count; i < imported_tensors.size(); ++i) {
            if (imported_tensors[i]) {
                input_shapes_vec.push_back(imported_tensors[i]->shape());
            }
        }

        // Fallback: If no graph inputs, use all imported inputs
        if (graph_input_count == 0 && !imported_tensors.empty()) {
            input_shapes_vec.clear();
            for (const auto& tensor : imported_tensors) {
                if (tensor) {
                    input_shapes_vec.push_back(tensor->shape());
                }
            }
        }

        if (!all_inputs_ready) {
            MI_LOG_ERROR("[ShapeInferenceEngine] Node '" + node->name() +
                         "': Not all inputs ready for shape inference");
            return core::Status::ERROR_RUNTIME;
        }

        // Infer output shapes
        std::vector<core::Shape> output_shapes;
        auto infer_status = op->infer_shape(input_shapes_vec, output_shapes);

        if (infer_status != core::Status::SUCCESS || output_shapes.empty()) {
            MI_LOG_ERROR("[ShapeInferenceEngine] Node '" + node->name() +
                         "': Shape inference failed (status=" +
                         std::to_string(static_cast<int>(infer_status)) + ")");
            return core::Status::ERROR_RUNTIME;
        }

        // Store inferred shapes using node ID (O(1) write)
        size_t node_id = node->id();
        if (node_id < inferred_shapes_.size()) {
            inferred_shapes_[node_id] = output_shapes;
        }

        total_inferred++;

        if (verbose_) {
            MI_LOG_INFO("[ShapeInferenceEngine] Node '" + node->name() +
                        "': " + output_shapes[0].to_string());
        }
    }

    // Cache input shapes for comparison
    last_input_shapes_ = input_shapes;
    last_input_shapes_lookup_.clear();
    for (const auto& binding : input_shapes) {
        if (binding.node_id < sorted_nodes_.size() && sorted_nodes_[binding.node_id]) {
            last_input_shapes_lookup_[sorted_nodes_[binding.node_id]->name()] = binding.shape;
        }
    }

    if (verbose_) {
        MI_LOG_INFO("[ShapeInferenceEngine] Shape inference completed: " +
                    std::to_string(total_inferred) + " node(s) inferred");
    }

    return core::Status::SUCCESS;
}

const core::Shape* ShapeInferenceEngine::get_inferred_shape(const std::string& tensor_name) const {
    // Find node by name (only used during shape change handling, not hot path)
    auto node = graph_->get_node(tensor_name);
    if (!node) {
        return nullptr;
    }

    size_t node_id = node->id();
    if (node_id >= inferred_shapes_.size() || inferred_shapes_[node_id].empty()) {
        return nullptr;
    }

    // Return first output shape (for single-output nodes)
    // TODO: Support explicit output index for multi-output nodes
    return &inferred_shapes_[node_id][0];
}

bool ShapeInferenceEngine::shapes_changed(
    const std::unordered_map<std::string, core::Shape>& input_shapes) const {
    if (input_shapes.size() != last_input_shapes_lookup_.size()) {
        return true;
    }

    for (const auto& [name, shape] : input_shapes) {
        auto it = last_input_shapes_lookup_.find(name);
        if (it == last_input_shapes_lookup_.end()) {
            return true;
        }

        if (it->second != shape) {
            return true;
        }
    }

    return false;
}

bool ShapeInferenceEngine::shapes_changed(
    const std::vector<RuntimeInputShape>& input_shapes) const {
    if (input_shapes.size() != last_input_shapes_.size()) {
        return true;
    }
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        if (input_shapes[i].node_id != last_input_shapes_[i].node_id) {
            return true;
        }
        if (input_shapes[i].shape != last_input_shapes_[i].shape) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> ShapeInferenceEngine::get_tensors_needing_reallocation() const {
    std::vector<std::string> tensors_to_reallocate;

    // Check all nodes in the graph
    for (const auto& node : sorted_nodes_) {
        size_t node_id = node->id();
        if (node_id >= inferred_shapes_.size() || inferred_shapes_[node_id].empty()) {
            continue;
        }

        // Check output tensors
        const auto& outputs = node->output_tensors();
        const auto& inferred_outputs = inferred_shapes_[node_id];

        for (size_t i = 0; i < outputs.size() && i < inferred_outputs.size(); ++i) {
            if (!outputs[i]) {
                continue;
            }

            // Compare with inferred shape (direct access via node ID)
            if (outputs[i]->shape() != inferred_outputs[i]) {
                tensors_to_reallocate.push_back(node->name());
            }
        }
    }

    return tensors_to_reallocate;
}

void ShapeInferenceEngine::clear_cache() {
    inferred_shapes_.clear();
    last_input_shapes_.clear();
    last_input_shapes_lookup_.clear();
}

}  // namespace runtime
}  // namespace mini_infer
