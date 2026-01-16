#include "mini_infer/runtime/shape_inference_engine.h"

#include <algorithm>
#include <cstring>

#include "mini_infer/runtime/shape_tensor_evaluator.h"
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
            auto node = graph_->get_node(binding.node_id);
            if (node) {
                MI_LOG_INFO("[ShapeInferenceEngine]   " + node->name() + ": " +
                            binding.shape.to_string());
            }
        }
    }

    // Resize storage to fit all node ids
    inferred_shapes_.clear();
    inferred_shapes_.resize(graph_->node_capacity());

    // Shape constants: store computed tensor values for shape-related ops
    std::unordered_map<size_t, std::shared_ptr<core::Tensor>> shape_constants;

    // TensorRT-style: evaluate shape tensors symbolically
    ShapeTensorEvaluator shape_evaluator;
    shape_evaluator.initialize(graph_);
    shape_evaluator.seed_from_initializers();
    shape_evaluator.set_known_output_shapes(&inferred_shapes_);

    std::unordered_map<std::string, core::Shape> input_shapes_map;

    // Store input shapes (by node ID)
    for (const auto& binding : input_shapes) {
        if (binding.node_id < inferred_shapes_.size()) {
            inferred_shapes_[binding.node_id] = {binding.shape};  // Input nodes have single output
        }
        auto node = graph_->get_node(binding.node_id);
        if (node) {
            input_shapes_map[node->name()] = binding.shape;
        }
    }

    shape_evaluator.evaluate(input_shapes_map);

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

        // Determine total number of inputs from both graph edges and imported tensors
        const auto& input_edges = node->inputs();
        const auto& imported_tensors = node->input_tensors();

        // Find max input port from edges
        int max_edge_port = -1;
        for (const auto& edge : input_edges) {
            max_edge_port = std::max(max_edge_port, edge.dst_port);
        }

        // Total inputs = max of (edge ports + 1) and imported_tensors size
        size_t total_inputs = std::max(
            max_edge_port >= 0 ? static_cast<size_t>(max_edge_port + 1) : 0,
            imported_tensors.size());

        if (total_inputs == 0 && imported_tensors.empty()) {
            // No inputs at all - skip this node or use fallback
            continue;
        }

        input_shapes_vec.resize(total_inputs);

        // Step 1: Fill from imported tensors first (weights, constants)
        for (size_t i = 0; i < imported_tensors.size(); ++i) {
            if (imported_tensors[i] && i < input_shapes_vec.size()) {
                input_shapes_vec[i] = imported_tensors[i]->shape();
            }
        }

        // Step 2: Override with shapes from graph connections (data tensors)
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

        // Check all inputs have valid shapes
        if (all_inputs_ready) {
            for (const auto& shape : input_shapes_vec) {
                if (shape.ndim() == 0) {
                    all_inputs_ready = false;
                    break;
                }
            }
        }

        if (!all_inputs_ready) {
            MI_LOG_ERROR("[ShapeInferenceEngine] Node '" + node->name() +
                         "': Not all inputs ready for shape inference");
            return core::Status::ERROR_RUNTIME;
        }

        // Infer output shapes using cached plugin
        std::vector<core::Shape> output_shapes;
        std::vector<core::DataType> output_dtypes;
        core::Status infer_status;

        auto* plugin = op->cached_plugin();
        if (plugin) {
            // Collect input dtypes
            std::vector<core::DataType> input_dtypes_vec;
            for (size_t i = 0; i < imported_tensors.size(); ++i) {
                if (imported_tensors[i]) {
                    input_dtypes_vec.push_back(imported_tensors[i]->dtype());
                } else {
                    input_dtypes_vec.push_back(core::DataType::FLOAT32);
                }
            }
            // Ensure input_dtypes_vec has same size as input_shapes_vec
            while (input_dtypes_vec.size() < input_shapes_vec.size()) {
                input_dtypes_vec.push_back(core::DataType::FLOAT32);
            }

            // Build input_tensors_for_inference: combine imported_tensors with shape_constants
            std::vector<std::shared_ptr<core::Tensor>> input_tensors_for_inference(total_inputs);

            // Step 1: Fill from graph edges (shape evaluator > shape_constants > output_tensors)
            for (const auto& edge : input_edges) {
                if (!edge.node || edge.dst_port < 0 || edge.src_port < 0) {
                    continue;
                }
                const size_t dst_index = static_cast<size_t>(edge.dst_port);
                if (dst_index >= input_tensors_for_inference.size()) {
                    continue;
                }

                // Prefer evaluated shape tensors
                auto* shape_value = shape_evaluator.get_shape_value(edge.node->id());
                if (shape_value && shape_value->is_valid) {
                    core::Shape sv_shape({static_cast<int64_t>(shape_value->data.size())});
                    auto sv_tensor = std::make_shared<core::Tensor>(
                        sv_shape, shape_value->dtype, core::DeviceType::CPU);
                    if (shape_value->dtype == core::DataType::INT64) {
                        std::memcpy(sv_tensor->data(), shape_value->data.data(),
                                    shape_value->data.size() * sizeof(int64_t));
                    } else if (shape_value->dtype == core::DataType::INT32) {
                        int32_t* dst = static_cast<int32_t*>(sv_tensor->data());
                        for (size_t i = 0; i < shape_value->data.size(); ++i) {
                            dst[i] = static_cast<int32_t>(shape_value->data[i]);
                        }
                    }
                    input_tensors_for_inference[dst_index] = sv_tensor;
                    continue;
                }

                // Fall back to shape constants computed by plugins
                auto it = shape_constants.find(edge.node->id());
                if (it != shape_constants.end() && it->second) {
                    input_tensors_for_inference[dst_index] = it->second;
                    continue;
                }

                // Fall back to output tensors
                const auto& outputs = edge.node->output_tensors();
                const size_t src_index = static_cast<size_t>(edge.src_port);
                if (src_index < outputs.size() && outputs[src_index]) {
                    input_tensors_for_inference[dst_index] = outputs[src_index];
                }
            }

            // Step 2: Fill from imported tensors
            for (size_t i = 0; i < imported_tensors.size() && i < total_inputs; ++i) {
                if (!input_tensors_for_inference[i]) {
                    input_tensors_for_inference[i] = imported_tensors[i];
                }
            }

            // Use infer_output_shapes_with_tensors to support dynamic parameters
            infer_status = plugin->infer_output_shapes_with_tensors(
                input_shapes_vec, input_dtypes_vec, input_tensors_for_inference,
                output_shapes, output_dtypes);

            // Shape-const evaluation: compute values for shape-related operators
            if (infer_status == core::Status::SUCCESS && !output_shapes.empty()) {
                bool is_shape_op = (node->type() == core::OpType::kSHAPE ||
                                    node->type() == core::OpType::kGATHER ||
                                    node->type() == core::OpType::kUNSQUEEZE ||
                                    node->type() == core::OpType::kSQUEEZE ||
                                    node->type() == core::OpType::kCONCAT ||
                                    node->type() == core::OpType::kCAST ||
                                    node->type() == core::OpType::kSLICE ||
                                    node->type() == core::OpType::kCONSTANT_OF_SHAPE);

                bool can_compute = false;
                if (is_shape_op) {
                    if (node->type() == core::OpType::kSHAPE) {
                        can_compute = !input_shapes_vec.empty() && input_shapes_vec[0].ndim() > 0;
                    } else {
                        can_compute = true;
                        for (const auto& t : input_tensors_for_inference) {
                            if (!t || !t->data()) {
                                can_compute = false;
                                break;
                            }
                        }
                    }
                }

                if (can_compute) {
                    auto shape_tensor = std::make_shared<core::Tensor>(
                        output_shapes[0],
                        output_dtypes.empty() ? core::DataType::INT64 : output_dtypes[0],
                        core::DeviceType::CPU);

                    std::vector<std::shared_ptr<core::Tensor>> exec_outputs = {shape_tensor};
                    std::vector<std::shared_ptr<core::Tensor>> exec_inputs = input_tensors_for_inference;

                    if (node->type() == core::OpType::kSHAPE) {
                        auto dummy = std::make_shared<core::Tensor>(
                            input_shapes_vec[0], core::DataType::FLOAT32, core::DeviceType::CPU);
                        exec_inputs = {dummy};
                    }

                    operators::PluginContext ctx;
                    ctx.device_context = nullptr;
                    auto exec_status = plugin->enqueue(exec_inputs, exec_outputs, ctx);

                    if (exec_status == core::Status::SUCCESS) {
                        shape_constants[node->id()] = shape_tensor;
                    }
                }
            }
        } else {
            MI_LOG_ERROR("[ShapeInferenceEngine] Node '" + node->name() +
                         "': No plugin available for shape inference");
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

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

        // Update shape evaluator with newly inferred shapes
        if (!output_shapes.empty() && output_shapes[0].ndim() > 0) {
            input_shapes_map[node->name()] = output_shapes[0];
            shape_evaluator.evaluate(input_shapes_map);
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
        auto node = graph_->get_node(binding.node_id);
        if (node) {
            last_input_shapes_lookup_[node->name()] = binding.shape;
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

const std::vector<core::Shape>* ShapeInferenceEngine::get_inferred_shapes(size_t node_id) const {
    if (node_id >= inferred_shapes_.size() || inferred_shapes_[node_id].empty()) {
        return nullptr;
    }
    return &inferred_shapes_[node_id];
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

void ShapeInferenceEngine::seed_input_shapes(
    const std::vector<RuntimeInputShape>& input_shapes) {
    last_input_shapes_ = input_shapes;
    last_input_shapes_lookup_.clear();
    if (!graph_) {
        return;
    }
    for (const auto& binding : input_shapes) {
        auto node = graph_->get_node(binding.node_id);
        if (node) {
            last_input_shapes_lookup_[node->name()] = binding.shape;
        }
    }
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
