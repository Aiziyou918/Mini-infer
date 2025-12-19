#include "mini_infer/runtime/inference_plan.h"

#include <algorithm>

#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/runtime/execution_context.h"
#include "mini_infer/utils/logger.h"

namespace mini_infer {
namespace runtime {

InferencePlan::InferencePlan(const EngineConfig& config) : config_(config) {}

core::Status InferencePlan::build(std::shared_ptr<graph::Graph> graph) {
    if (!graph) {
        MI_LOG_ERROR("[InferencePlan] Graph is null");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    graph_ = graph;

    MI_LOG_INFO("[InferencePlan] ========================================");
    MI_LOG_INFO("[InferencePlan] Building Inference Plan");
    MI_LOG_INFO("[InferencePlan] ========================================");

    // Step 1: Graph optimization (operator fusion, constant folding, etc.)
    if (config_.enable_graph_optimization) {
        MI_LOG_INFO("[InferencePlan] Step 1: Applying graph optimizations...");
        auto status = optimize_graph();
        if (status != core::Status::SUCCESS) {
            MI_LOG_WARNING("[InferencePlan] Graph optimization failed, using original graph");
        }
    } else {
        MI_LOG_INFO("[InferencePlan] Step 1: Graph optimization disabled");
    }

    // Step 2: Topological sort with validation
    MI_LOG_INFO("[InferencePlan] Step 2: Performing topological sort...");
    auto status = graph_->checked_topological_sort(sorted_nodes_);
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("[InferencePlan] Topological sort failed");
        return status;
    }
    MI_LOG_INFO("[InferencePlan] Topological sort completed: " +
                std::to_string(sorted_nodes_.size()) + " nodes");

    // Step 2.5: Assign unique IDs to nodes for fast runtime indexing
    MI_LOG_INFO("[InferencePlan] Step 2.5: Assigning node IDs...");
    for (size_t i = 0; i < sorted_nodes_.size(); ++i) {
        if (sorted_nodes_[i]) {
            sorted_nodes_[i]->set_id(i);
        }
    }
    MI_LOG_INFO("[InferencePlan] Node ID assignment completed");

    auto binding_status = initialize_input_bindings();
    if (binding_status != core::Status::SUCCESS) {
        MI_LOG_ERROR("[InferencePlan] Failed to initialize input bindings");
        return binding_status;
    }

    // Step 3: Shape inference
    MI_LOG_INFO("[InferencePlan] Step 3: Inferring tensor shapes...");
    if (config_.enable_dynamic_shapes && config_.optimization_profile) {
        // Use max shapes from profile for build-time inference
        status = infer_shapes_with_profile();
    } else {
        // Traditional static shape inference
        status = infer_shapes();
    }
    if (status != core::Status::SUCCESS) {
        MI_LOG_WARNING("[InferencePlan] Shape inference incomplete");
    }

    // Step 3.5: Update tensor metadata (shape/dtype/size) before memory planning
    MI_LOG_INFO("[InferencePlan] Step 3.5: Updating tensor metadata (shape/dtype/size)...");
    status = update_tensor_properties();
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("[InferencePlan] Failed to update tensor metadata");
        return status;
    }

    // Step 4: Memory planning (TensorRT-style static memory allocation)
    if (config_.enable_memory_planning) {
        MI_LOG_INFO("[InferencePlan] Step 4: Planning memory allocation...");
        status = plan_memory();
        if (status != core::Status::SUCCESS) {
            MI_LOG_WARNING("[InferencePlan] Memory planning failed, using default allocation");
        }
    } else {
        MI_LOG_INFO("[InferencePlan] Step 4: Memory planning disabled");
    }

    MI_LOG_INFO("[InferencePlan] ========================================");
    MI_LOG_INFO("[InferencePlan] Inference Plan built successfully");
    MI_LOG_INFO("[InferencePlan] ========================================");
    return core::Status::SUCCESS;
}

std::shared_ptr<ExecutionContext> InferencePlan::create_execution_context() const {
    try {
        auto ctx = std::make_shared<ExecutionContext>(shared_from_this());
        auto status = ctx->initialize();
        if (status != core::Status::SUCCESS) {
            MI_LOG_ERROR("[InferencePlan] Failed to initialize execution context");
            return nullptr;
        }
        return ctx;
    } catch (const std::bad_weak_ptr&) {
        MI_LOG_ERROR("[InferencePlan] create_execution_context requires shared ownership");
        return nullptr;
    }
}

core::Status InferencePlan::execute(ExecutionContext* ctx) const {
    if (!graph_) {
        return core::Status::ERROR_RUNTIME;
    }
    if (!ctx) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    if (!ctx->initialized_) {
        auto status = ctx->initialize();
        if (status != core::Status::SUCCESS) {
            return status;
        }
    }

    if (input_bindings_.empty()) {
        return core::Status::ERROR_RUNTIME;
    }

    if (ctx->ordered_inputs_.size() != input_bindings_.size()) {
        MI_LOG_ERROR("[InferencePlan] Expected " + std::to_string(input_bindings_.size()) +
                     " inputs, but got " + std::to_string(ctx->ordered_inputs_.size()));
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    std::vector<ShapeInferenceEngine::RuntimeInputShape> runtime_input_shapes;
    auto status = build_runtime_shapes(ctx->ordered_inputs_, runtime_input_shapes);
    if (status != core::Status::SUCCESS) {
        return status;
    }

    if (config_.enable_dynamic_shapes && ctx->shape_inference_engine_) {
        if (check_shape_change(ctx, runtime_input_shapes)) {
            MI_LOG_INFO("[InferencePlan] Input shape changed, re-inferring shapes...");
            status = handle_shape_change(ctx, runtime_input_shapes);
            if (status != core::Status::SUCCESS) {
                MI_LOG_ERROR("[InferencePlan] Failed to handle shape change");
                return status;
            }
        }
    }

    status = bind_ordered_inputs(ctx, ctx->ordered_inputs_);
    if (status != core::Status::SUCCESS) {
        return status;
    }

    // Execute all nodes (skip placeholder nodes without operator)
    for (const auto& node : sorted_nodes_) {
        if (!node || !node->get_operator()) {
            continue;
        }
        auto exec_status = ctx->execute_node(node);
        if (exec_status != core::Status::SUCCESS) {
            MI_LOG_ERROR("Node execution failed: " + node->name());
            return exec_status;
        }
    }

    return collect_ordered_outputs(ctx);
}

core::Status InferencePlan::optimize_graph() {
    // TensorRT-style: Use registered optimization passes
    auto optimizer = graph::GraphOptimizer::create_default();
    optimizer.set_verbose(config_.enable_profiling);

    auto status = optimizer.optimize(graph_.get());
    optimization_stats_ = optimizer.get_statistics();

    if (status == core::Status::SUCCESS) {
        MI_LOG_INFO("[InferencePlan] Graph optimization completed: " +
                    std::to_string(optimization_stats_.total_modifications) + " modification(s)");
    }

    return status;
}

core::Status InferencePlan::infer_shapes() {
    int total_inferred = 0;

    for (const auto& node : sorted_nodes_) {
        if (!node) {
            continue;
        }

        if (!node->get_operator()) {
            continue;
        }

        std::vector<core::Shape> input_shapes;
        const auto& input_edges = node->inputs();
        size_t graph_input_count = 0;
        if (!input_edges.empty()) {
            int max_dst_port = -1;
            for (const auto& edge : input_edges) {
                max_dst_port = std::max(max_dst_port, edge.dst_port);
            }
            graph_input_count = static_cast<size_t>(max_dst_port + 1);
            input_shapes.resize(graph_input_count);
        }

        for (const auto& edge : input_edges) {
            if (!edge.node || edge.dst_port < 0 || edge.src_port < 0) {
                continue;
            }
            const auto& outputs = edge.node->output_tensors();
            const size_t src_index = static_cast<size_t>(edge.src_port);
            const size_t dst_index = static_cast<size_t>(edge.dst_port);
            if (src_index >= outputs.size() || !outputs[src_index]) {
                continue;
            }
            if (dst_index >= input_shapes.size()) {
                continue;
            }
            input_shapes[dst_index] = outputs[src_index]->shape();
        }

        const auto& imported_inputs = node->input_tensors();
        for (size_t i = graph_input_count; i < imported_inputs.size(); ++i) {
            if (imported_inputs[i]) {
                input_shapes.push_back(imported_inputs[i]->shape());
            }
        }

        if (graph_input_count == 0 && !imported_inputs.empty()) {
            input_shapes.clear();
            for (const auto& tensor : imported_inputs) {
                if (tensor) {
                    input_shapes.push_back(tensor->shape());
                }
            }
        }

        if (input_shapes.empty()) {
            MI_LOG_ERROR("[InferencePlan] Node " + node->name() +
                         " has no input shapes, cannot infer output shape");
            return core::Status::ERROR_RUNTIME;
        }

        std::vector<core::Shape> output_shapes;
        auto status = node->get_operator()->infer_shape(input_shapes, output_shapes);
        if (status != core::Status::SUCCESS) {
            MI_LOG_ERROR("[InferencePlan] Failed to infer shape for node: " + node->name() +
                         " (status=" + std::to_string(static_cast<int>(status)) + ")");
            return status;
        }

        if (output_shapes.empty()) {
            MI_LOG_ERROR("[InferencePlan] Node " + node->name() + " produced empty output shapes");
            return core::Status::ERROR_RUNTIME;
        }

        auto& output_tensors = node->output_tensors();
        while (output_tensors.size() < output_shapes.size()) {
            output_tensors.push_back(std::make_shared<core::Tensor>());
        }

        for (size_t i = 0; i < output_shapes.size() && i < output_tensors.size(); ++i) {
            if (!output_tensors[i]) {
                output_tensors[i] = std::make_shared<core::Tensor>();
            }
            output_tensors[i]->set_shape_metadata(output_shapes[i]);

            if (output_tensors[i]->dtype() == core::DataType::FLOAT32 &&
                !node->input_tensors().empty() && node->input_tensors()[0]) {
                output_tensors[i]->set_dtype(node->input_tensors()[0]->dtype());
            }

            total_inferred++;

            if (config_.enable_profiling) {
                MI_LOG_INFO("[InferencePlan] Node " + node->name() + " output[" +
                            std::to_string(i) + "] shape: " + output_shapes[i].to_string());
            }
        }
    }

    MI_LOG_INFO("[InferencePlan] Shape inference completed: " +
                std::to_string(total_inferred) + " tensor(s) inferred");

    return core::Status::SUCCESS;
}

core::Status InferencePlan::infer_shapes_with_profile() {
    if (!config_.optimization_profile) {
        MI_LOG_ERROR("[InferencePlan] Optimization profile is null");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto max_shapes = config_.optimization_profile->get_max_shapes();

    if (config_.enable_profiling) {
        MI_LOG_INFO("[InferencePlan] Using max profile with optimal shapes:");
        for (const auto& [name, shape] : max_shapes) {
            MI_LOG_INFO("[InferencePlan]   " + name + ": " + shape.to_string());
        }
    }

    for (const auto& input_name : graph_->inputs()) {
        auto it = max_shapes.find(input_name);
        if (it == max_shapes.end()) {
            MI_LOG_WARNING("[InferencePlan] No max shape for input '" + input_name + "'");
            continue;
        }

        auto node = graph_->get_node(input_name);
        if (node) {
            if (node->output_tensors().empty() || !node->output_tensors()[0]) {
                auto tensor = std::make_shared<core::Tensor>(it->second, core::DataType::FLOAT32);
                node->set_output_tensors({tensor});
            } else {
                auto tensor = std::make_shared<core::Tensor>(it->second,
                                                            node->output_tensors()[0]->dtype());
                node->set_output_tensors({tensor});
            }

            if (config_.enable_profiling) {
                MI_LOG_INFO("[InferencePlan] Set input '" + input_name +
                            "' shape: " + it->second.to_string());
            }
        }
    }

    return infer_shapes();
}

core::Status InferencePlan::update_tensor_properties() {
    if (!graph_) {
        MI_LOG_ERROR("[InferencePlan] Graph is null");
        return core::Status::ERROR_RUNTIME;
    }

    size_t updated_count = 0;

    for (const auto& node : sorted_nodes_) {
        if (!node) {
            continue;
        }

        auto& outputs = node->output_tensors();
        for (size_t idx = 0; idx < outputs.size(); ++idx) {
            auto& tensor = outputs[idx];
            if (!tensor) {
                continue;
            }

            const auto& shape = tensor->shape();
            if (shape.ndim() == 0) {
                MI_LOG_ERROR("[InferencePlan] Tensor '" + node->name() + "' output[" +
                             std::to_string(idx) + "] has undefined shape");
                return core::Status::ERROR_RUNTIME;
            }

            bool has_dynamic_dim = false;
            for (size_t d = 0; d < shape.ndim(); ++d) {
                if (shape[d] < 0) {
                    has_dynamic_dim = true;
                    break;
                }
            }

            if (has_dynamic_dim) {
                if (config_.enable_dynamic_shapes) {
                    MI_LOG_ERROR("[InferencePlan] Tensor '" + node->name() + "' output[" +
                                 std::to_string(idx) + "] still has dynamic dimensions " +
                                 "after shape inference. Provide an OptimizationProfile " +
                                 "or disable dynamic shapes.");
                    return core::Status::ERROR_INVALID_ARGUMENT;
                }

                std::vector<int64_t> concrete_dims(shape.dims().begin(), shape.dims().end());
                for (auto& dim : concrete_dims) {
                    if (dim < 0) {
                        dim = 1;
                    }
                }

                tensor->set_shape_metadata(core::Shape(concrete_dims));
                if (config_.enable_profiling) {
                    MI_LOG_WARNING("[InferencePlan] Tensor '" + node->name() + "' output[" +
                                   std::to_string(idx) +
                                   "] had dynamic dimensions; defaulting to batch=1 shape " +
                                   tensor->shape().to_string());
                }
            }

            auto inherit_dtype_from_tensor =
                [&tensor](const std::shared_ptr<core::Tensor>& src) -> bool {
                if (!src)
                    return false;
                tensor->set_dtype(src->dtype());
                return true;
            };

            if (tensor->dtype() == core::DataType::FLOAT32) {
                bool dtype_set = false;
                for (const auto& edge : node->inputs()) {
                    if (edge.node && !edge.node->output_tensors().empty()) {
                        dtype_set = inherit_dtype_from_tensor(edge.node->output_tensors()[0]);
                        if (dtype_set)
                            break;
                    }
                }
                if (!dtype_set) {
                    for (const auto& imported : node->input_tensors()) {
                        if (inherit_dtype_from_tensor(imported)) {
                            break;
                        }
                    }
                }
            }

            const int64_t numel = tensor->shape().numel();
            if (numel <= 0) {
                MI_LOG_ERROR("[InferencePlan] Tensor '" + node->name() + "' output[" +
                             std::to_string(idx) + "] has invalid numel=" + std::to_string(numel));
                return core::Status::ERROR_RUNTIME;
            }

            const size_t size_bytes = tensor->size_in_bytes();
            if (size_bytes == 0 && numel > 0) {
                MI_LOG_ERROR("[InferencePlan] Tensor '" + node->name() + "' output[" +
                             std::to_string(idx) + "] size_in_bytes()=0 (shape=" +
                             tensor->shape().to_string() + ")");
                return core::Status::ERROR_RUNTIME;
            }

            updated_count++;

            if (config_.enable_profiling) {
                MI_LOG_INFO("[InferencePlan] Tensor '" + node->name() + "' output[" +
                            std::to_string(idx) + "]: shape=" + shape.to_string() +
                            ", dtype=" + std::to_string(static_cast<int>(tensor->dtype())) +
                            ", size=" + std::to_string(size_bytes) + " bytes");
            }
        }
    }

    MI_LOG_INFO("[InferencePlan] Updated metadata for " + std::to_string(updated_count) +
                " tensor(s)");
    return core::Status::SUCCESS;
}

core::Status InferencePlan::plan_memory() {
    MemoryPlanner planner;
    planner.set_enabled(true);
    planner.set_verbose(config_.enable_profiling);
    planner.set_alignment(config_.memory_alignment);

    memory_plan_ = planner.plan(graph_.get());

    if (memory_plan_.pools.empty()) {
        MI_LOG_WARNING("[InferencePlan] Memory planning produced no pools");
        return core::Status::ERROR_RUNTIME;
    }

    MI_LOG_INFO("[InferencePlan] Memory planning completed:");
    MI_LOG_INFO("[InferencePlan]   Original memory:  " +
                std::to_string(memory_plan_.original_memory / 1024.0) + " KB");
    MI_LOG_INFO("[InferencePlan]   Optimized memory: " +
                std::to_string(memory_plan_.total_memory / 1024.0) + " KB");
    MI_LOG_INFO("[InferencePlan]   Memory saving:    " +
                std::to_string(memory_plan_.memory_saving_ratio * 100.0f) + "%");
    MI_LOG_INFO("[InferencePlan]   Number of pools:  " +
                std::to_string(memory_plan_.pools.size()));

    return core::Status::SUCCESS;
}

core::Status InferencePlan::initialize_input_bindings() {
    input_bindings_.clear();

    if (!graph_) {
        return core::Status::ERROR_RUNTIME;
    }

    const auto& input_names = graph_->inputs();
    input_bindings_.reserve(input_names.size());

    for (const auto& name : input_names) {
        auto node = graph_->get_node(name);
        if (!node) {
            MI_LOG_ERROR("[InferencePlan] Graph input node not found: " + name);
            input_bindings_.clear();
            return core::Status::ERROR_RUNTIME;
        }

        InputBinding binding;
        binding.name = name;
        binding.node_id = node->id();
        binding.node = node.get();
        input_bindings_.push_back(binding);
    }

    return core::Status::SUCCESS;
}

core::Status InferencePlan::gather_map_inputs(
    const std::unordered_map<std::string, std::shared_ptr<core::Tensor>>& inputs,
    std::vector<std::shared_ptr<core::Tensor>>& ordered_inputs) const {
    if (!graph_) {
        return core::Status::ERROR_RUNTIME;
    }

    ordered_inputs.clear();
    const auto& input_names = graph_->inputs();
    ordered_inputs.reserve(input_names.size());

    for (const auto& name : input_names) {
        auto it = inputs.find(name);
        if (it == inputs.end() || !it->second) {
            MI_LOG_ERROR("[InferencePlan] Missing input: " + name);
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
        ordered_inputs.push_back(it->second);
    }

    return core::Status::SUCCESS;
}

core::Status InferencePlan::build_runtime_shapes(
    const std::vector<std::shared_ptr<core::Tensor>>& ordered_inputs,
    std::vector<ShapeInferenceEngine::RuntimeInputShape>& runtime_shapes) const {
    runtime_shapes.clear();
    runtime_shapes.reserve(input_bindings_.size());

    for (size_t idx = 0; idx < input_bindings_.size(); ++idx) {
        const auto& tensor = ordered_inputs[idx];
        if (!tensor) {
            MI_LOG_ERROR("[InferencePlan] Input tensor at index " + std::to_string(idx) +
                         " is null");
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        ShapeInferenceEngine::RuntimeInputShape runtime_shape;
        runtime_shape.node_id = input_bindings_[idx].node_id;
        runtime_shape.shape = tensor->shape();
        runtime_shapes.push_back(runtime_shape);
    }

    return core::Status::SUCCESS;
}

core::Status InferencePlan::bind_ordered_inputs(
    ExecutionContext* ctx, const std::vector<std::shared_ptr<core::Tensor>>& ordered_inputs) const {
    for (size_t idx = 0; idx < input_bindings_.size(); ++idx) {
        const auto& binding = input_bindings_[idx];
        if (!binding.node) {
            MI_LOG_ERROR("[InferencePlan] Input binding node is null for: " + binding.name);
            return core::Status::ERROR_RUNTIME;
        }

        const auto& tensor = ordered_inputs[idx];
        if (!tensor) {
            MI_LOG_ERROR("[InferencePlan] Input tensor is null for binding: " + binding.name);
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        if (binding.node_id >= ctx->node_outputs_.size()) {
            MI_LOG_ERROR("[InferencePlan] Input binding node id out of range for: " +
                         binding.name);
            return core::Status::ERROR_RUNTIME;
        }

        auto& outputs = ctx->node_outputs_[binding.node_id];
        if (outputs.empty()) {
            outputs.resize(1);
        }

        if (!binding.node->output_tensors().empty() && binding.node->output_tensors()[0]) {
            const auto& expected_shape = binding.node->output_tensors()[0]->shape();
            const auto& actual_shape = tensor->shape();

            if (expected_shape.ndim() > 0 && expected_shape.ndim() == actual_shape.ndim()) {
                bool compatible = true;
                for (size_t i = 0; i < expected_shape.ndim(); ++i) {
                    if (expected_shape[i] < 0 || i == 0)
                        continue;

                    if (expected_shape[i] != actual_shape[i]) {
                        MI_LOG_ERROR("[InferencePlan] Input '" + binding.name +
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

        outputs[0] = tensor;
    }

    return core::Status::SUCCESS;
}

core::Status InferencePlan::collect_ordered_outputs(ExecutionContext* ctx) const {
    if (!graph_) {
        return core::Status::ERROR_RUNTIME;
    }

    ctx->ordered_outputs_.clear();
    ctx->named_outputs_.clear();
    ctx->tensor_map_.clear();

    const auto& output_names = graph_->outputs();
    ctx->ordered_outputs_.reserve(output_names.size());

    for (const auto& output_name : output_names) {
        auto node = graph_->get_node(output_name);
        if (!node) {
            MI_LOG_ERROR("[InferencePlan] Output node not found: " + output_name);
            return core::Status::ERROR_RUNTIME;
        }

        std::shared_ptr<core::Tensor> output_tensor;
        if (node->id() < ctx->node_outputs_.size() && !ctx->node_outputs_[node->id()].empty()) {
            output_tensor = ctx->node_outputs_[node->id()][0];
        }

        ctx->ordered_outputs_.push_back(output_tensor);
        if (output_tensor) {
            ctx->named_outputs_[output_name] = output_tensor;
        }
    }

    for (const auto& node : sorted_nodes_) {
        if (!node) {
            continue;
        }
        if (node->id() >= ctx->node_outputs_.size()) {
            continue;
        }
        const auto& outputs = ctx->node_outputs_[node->id()];
        if (!outputs.empty() && outputs[0]) {
            ctx->tensor_map_[node->name()] = outputs[0];
        }
    }

    return core::Status::SUCCESS;
}

bool InferencePlan::check_shape_change(
    ExecutionContext* ctx,
    const std::vector<ShapeInferenceEngine::RuntimeInputShape>& runtime_shapes) const {
    if (!ctx->shape_inference_engine_) {
        return false;
    }
    return ctx->shape_inference_engine_->shapes_changed(runtime_shapes);
}

core::Status InferencePlan::handle_shape_change(
    ExecutionContext* ctx,
    const std::vector<ShapeInferenceEngine::RuntimeInputShape>& runtime_shapes) const {
    if (config_.optimization_profile) {
        if (runtime_shapes.size() != input_bindings_.size()) {
            MI_LOG_ERROR("[InferencePlan] Input binding count mismatch during profile validation");
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
        for (size_t idx = 0; idx < input_bindings_.size(); ++idx) {
            const auto& binding = input_bindings_[idx];
            const auto* range = config_.optimization_profile->get_shape_range(binding.name);
            if (!range) {
                continue;
            }
            if (!range->contains(runtime_shapes[idx].shape)) {
                MI_LOG_ERROR("[InferencePlan] Input '" + binding.name +
                             "' shape is outside optimization profile range");
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
        }
    }

    auto status = ctx->shape_inference_engine_->infer_shapes(runtime_shapes);
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("[InferencePlan] Runtime shape inference failed");
        return status;
    }

    size_t resized = 0;
    for (const auto& node : sorted_nodes_) {
        if (!node) {
            continue;
        }
        auto inferred = ctx->shape_inference_engine_->get_inferred_shape(node->name());
        if (!inferred) {
            continue;
        }

        if (node->id() >= ctx->node_outputs_.size()) {
            continue;
        }
        auto& outputs = ctx->node_outputs_[node->id()];
        if (outputs.empty() || !outputs[0]) {
            continue;
        }

        if (outputs[0]->shape() != *inferred) {
            outputs[0]->resize(*inferred);
            resized++;

            if (config_.enable_profiling) {
                MI_LOG_INFO("[InferencePlan]   Resized '" + node->name() +
                            "': " + inferred->to_string());
            }
        }
    }

    if (resized > 0) {
        MI_LOG_INFO("[InferencePlan] Resized " + std::to_string(resized) +
                    " tensor(s) due to shape change");
    }

    return core::Status::SUCCESS;
}

std::vector<std::string> InferencePlan::get_input_names() const {
    if (graph_) {
        return graph_->inputs();
    }
    return {};
}

std::vector<std::string> InferencePlan::get_output_names() const {
    if (graph_) {
        return graph_->outputs();
    }
    return {};
}

}  // namespace runtime
}  // namespace mini_infer
