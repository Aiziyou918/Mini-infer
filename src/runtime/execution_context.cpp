#include "mini_infer/runtime/execution_context.h"

#include <algorithm>
#include <cstring>

#include "mini_infer/core/allocator.h"
#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/runtime/inference_plan.h"
#include "mini_infer/utils/logger.h"

namespace mini_infer {
namespace runtime {

ExecutionContext::ExecutionContext(std::shared_ptr<const InferencePlan> plan)
    : plan_(std::move(plan)) {}

core::Status ExecutionContext::initialize() {
    if (initialized_) {
        return core::Status::SUCCESS;
    }
    if (!plan_ || !plan_->graph()) {
        return core::Status::ERROR_RUNTIME;
    }

    contexts_.emplace(core::DeviceType::CPU, std::make_shared<backends::CPUDeviceContext>());
    node_outputs_.clear();
    node_outputs_.resize(plan_->graph()->node_capacity());

    auto status = allocate_tensors();
    if (status != core::Status::SUCCESS) {
        return status;
    }

    if (plan_->config().enable_dynamic_shapes) {
        shape_inference_engine_ = std::make_unique<ShapeInferenceEngine>(plan_->graph());
        shape_inference_engine_->set_verbose(plan_->config().enable_profiling);
    }

    initialized_ = true;
    return core::Status::SUCCESS;
}

core::Status ExecutionContext::set_inputs(
    const std::unordered_map<std::string, std::shared_ptr<core::Tensor>>& inputs) {
    if (!plan_) {
        return core::Status::ERROR_RUNTIME;
    }
    return plan_->gather_map_inputs(inputs, ordered_inputs_);
}

core::Status ExecutionContext::set_inputs(
    const std::vector<std::shared_ptr<core::Tensor>>& inputs) {
    ordered_inputs_ = inputs;
    return core::Status::SUCCESS;
}

void ExecutionContext::clear_io() {
    ordered_inputs_.clear();
    ordered_outputs_.clear();
    named_outputs_.clear();
    tensor_map_.clear();
}

core::Status ExecutionContext::allocate_tensors() {
    int allocated_count = 0;
    int skipped_count = 0;
    int failed_count = 0;

    const bool use_memory_pools =
        plan_->config().enable_memory_planning && !plan_->get_memory_plan().pools.empty();

    auto status = prepare_memory_pools(use_memory_pools);
    if (status != core::Status::SUCCESS) {
        return status;
    }

    for (const auto& node : plan_->sorted_nodes()) {
        status = allocate_node_outputs(node, use_memory_pools, allocated_count, skipped_count,
                                       failed_count);
        if (status != core::Status::SUCCESS) {
            return status;
        }
    }

    MI_LOG_INFO("[ExecutionContext] Tensor allocation completed: " +
                std::to_string(allocated_count) + " allocated, " +
                std::to_string(skipped_count) + " skipped, " +
                std::to_string(failed_count) + " failed");

    if (failed_count > 0) {
        MI_LOG_WARNING("[ExecutionContext] Some tensors failed to allocate, inference may fail");
    }

    return core::Status::SUCCESS;
}

core::Status ExecutionContext::prepare_memory_pools(bool use_memory_pools) {
    memory_pool_buffers_.clear();
    shared_buffer_.reset();
    shared_buffer_size_ = 0;

    if (!use_memory_pools) {
        if (plan_->config().enable_memory_planning && plan_->config().enable_profiling) {
            MI_LOG_WARNING(
                "[ExecutionContext] Memory planning enabled but no pools available; "
                "falling back to per-tensor allocations");
        }
        return core::Status::SUCCESS;
    }

    const auto& plan = plan_->get_memory_plan();
    if (plan.shared_buffer_size > 0) {
        shared_buffer_size_ = plan.shared_buffer_size;
        void* raw = core::CPUAllocator::get_instance()->allocate(
            shared_buffer_size_, plan_->config().memory_alignment);
        if (!raw) {
            MI_LOG_ERROR("[ExecutionContext] Failed to allocate shared buffer of size " +
                         std::to_string(shared_buffer_size_) + " bytes");
            return core::Status::ERROR_RUNTIME;
        }
        std::memset(raw, 0, shared_buffer_size_);
        shared_buffer_.reset(raw, [](void* p) {
            core::CPUAllocator::get_instance()->deallocate(p);
        });
        if (plan_->config().enable_profiling) {
            MI_LOG_INFO("[ExecutionContext] Created shared buffer (" +
                        std::to_string(shared_buffer_size_ / 1024.0) + " KB)");
        }
        return core::Status::SUCCESS;
    }

    const auto& pools = plan.pools;
    memory_pool_buffers_.reserve(pools.size());
    for (const auto& pool : pools) {
        void* raw = nullptr;
        if (pool.size_bytes > 0) {
            raw = core::CPUAllocator::get_instance()->allocate(
                pool.size_bytes, plan_->config().memory_alignment);
        }

        if (!raw && pool.size_bytes > 0) {
            MI_LOG_ERROR("[ExecutionContext] Failed to allocate memory pool " +
                         std::to_string(pool.pool_id) + " of size " +
                         std::to_string(pool.size_bytes) + " bytes");
            return core::Status::ERROR_RUNTIME;
        }

        if (raw) {
            std::memset(raw, 0, pool.size_bytes);
        }

        memory_pool_buffers_.emplace_back(
            raw, [](void* p) { core::CPUAllocator::get_instance()->deallocate(p); });

        if (plan_->config().enable_profiling) {
            MI_LOG_INFO("[ExecutionContext] Created memory pool " + std::to_string(pool.pool_id) +
                        " (" + std::to_string(pool.size_bytes / 1024.0) + " KB)");
        }
    }

    return core::Status::SUCCESS;
}

core::Status ExecutionContext::allocate_node_outputs(const std::shared_ptr<graph::Node>& node,
                                                     bool use_memory_pools, int& allocated_count,
                                                     int& skipped_count, int& failed_count) {
    if (!node) {
        return core::Status::SUCCESS;
    }
    if (!node->get_operator()) {
        return core::Status::SUCCESS;
    }

    if (node->id() >= node_outputs_.size()) {
        return core::Status::ERROR_RUNTIME;
    }

    const auto& template_outputs = node->output_tensors();
    auto& outputs = node_outputs_[node->id()];
    if (outputs.size() < template_outputs.size()) {
        outputs.resize(template_outputs.size());
    }

    for (size_t i = 0; i < template_outputs.size(); ++i) {
        const auto& tmpl = template_outputs[i];
        if (!tmpl) {
            failed_count++;
            continue;
        }

        auto& tensor = outputs[i];
        if (tensor && !tensor->empty()) {
            skipped_count++;
            continue;
        }

        const auto& shape = tmpl->shape();
        if (shape.ndim() == 0) {
            if (plan_->config().enable_profiling) {
                MI_LOG_WARNING("[ExecutionContext] Node " + node->name() + " output[" +
                               std::to_string(i) + "] has empty shape, skipping allocation");
            }
            failed_count++;
            continue;
        }

        if (shape.numel() <= 0) {
            MI_LOG_ERROR("[ExecutionContext] Node " + node->name() + " output[" +
                         std::to_string(i) + "] has invalid shape: " + shape.to_string());
            failed_count++;
            continue;
        }

        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
        }
        tensor->set_shape_metadata(shape);
        tensor->set_dtype(tmpl->dtype());

        const auto bind_result =
            try_bind_tensor_to_pool(node->id(), i, tensor, use_memory_pools, allocated_count,
                                    failed_count);
        if (bind_result == PoolBindResult::kBound) {
            continue;
        }
        if (bind_result == PoolBindResult::kFailed) {
            continue;
        }

        try {
            *tensor = core::Tensor(shape, tensor->dtype());
            allocated_count++;

            if (plan_->config().enable_profiling) {
                MI_LOG_INFO("[ExecutionContext] Allocated tensor for " + node->name() +
                            " output[" + std::to_string(i) + "]: " + shape.to_string() + " (" +
                            std::to_string(tensor->size_in_bytes() / 1024.0) + " KB)");
            }
        } catch (const std::exception& e) {
            MI_LOG_ERROR("[ExecutionContext] Failed to allocate tensor for " + node->name() +
                         " output[" + std::to_string(i) + "]: " + e.what());
            failed_count++;
        }
    }

    return core::Status::SUCCESS;
}

ExecutionContext::PoolBindResult ExecutionContext::try_bind_tensor_to_pool(
    size_t node_id, size_t output_index, std::shared_ptr<core::Tensor>& tensor,
    bool use_memory_pools, int& allocated_count, int& failed_count) {
    if (!use_memory_pools) {
        return PoolBindResult::kNotTried;
    }

    const auto& plan = plan_->get_memory_plan();
    if (node_id < plan.tensor_offsets.size() &&
        plan.tensor_offsets[node_id] != MemoryPlan::kInvalidOffset) {
        if (!shared_buffer_) {
            MI_LOG_ERROR("[ExecutionContext] Shared buffer not initialized for node " +
                         std::to_string(node_id));
            failed_count++;
            return PoolBindResult::kFailed;
        }

        const size_t required = tensor->size_in_bytes();
        const size_t offset = plan.tensor_offsets[node_id];
        if (offset + required > shared_buffer_size_) {
            MI_LOG_ERROR("[ExecutionContext] Node " + std::to_string(node_id) + " output[" +
                         std::to_string(output_index) + "] requires " +
                         std::to_string(required) + " bytes at offset " +
                         std::to_string(offset) + ", exceeds shared buffer size " +
                         std::to_string(shared_buffer_size_));
            failed_count++;
            return PoolBindResult::kFailed;
        }

        if (!tensor->bind_external_data_with_offset(shared_buffer_, shared_buffer_size_, offset)) {
            MI_LOG_ERROR("[ExecutionContext] Failed to bind shared buffer for node " +
                         std::to_string(node_id));
            failed_count++;
            return PoolBindResult::kFailed;
        }

        allocated_count++;
        if (plan_->config().enable_profiling) {
            MI_LOG_INFO("[ExecutionContext] Bound tensor for node " + std::to_string(node_id) +
                        " output[" +
                        std::to_string(output_index) + "] to shared buffer offset " +
                        std::to_string(offset) + " (" + std::to_string(required / 1024.0) +
                        " KB)");
        }
        return PoolBindResult::kBound;
    }

    if (node_id >= plan.tensor_to_pool.size() ||
        plan.tensor_to_pool[node_id] == MemoryPlan::kInvalidPool) {
        return PoolBindResult::kNotTried;
    }

    int pool_id = plan.tensor_to_pool[node_id];
    const bool valid_pool =
        pool_id >= 0 && static_cast<size_t>(pool_id) < memory_pool_buffers_.size() &&
        static_cast<size_t>(pool_id) < plan.pools.size() && memory_pool_buffers_[pool_id] != nullptr;

    if (!valid_pool) {
        MI_LOG_WARNING("[ExecutionContext] Memory plan pool unavailable for node " +
                       std::to_string(node_id) + ", falling back to independent allocation");
        return PoolBindResult::kNotTried;
    }

    const size_t required = tensor->size_in_bytes();
    const size_t pool_size = plan.pools[static_cast<size_t>(pool_id)].size_bytes;

    if (required > pool_size) {
        MI_LOG_ERROR("[ExecutionContext] Node " + std::to_string(node_id) + " output[" +
                     std::to_string(output_index) + "] requires " +
                     std::to_string(required) + " bytes, but pool " +
                     std::to_string(pool_id) + " size is " + std::to_string(pool_size));
        failed_count++;
        return PoolBindResult::kFailed;
    }

    tensor->bind_external_data(memory_pool_buffers_[pool_id],
                               plan.pools[static_cast<size_t>(pool_id)].size_bytes);
    allocated_count++;

    if (plan_->config().enable_profiling) {
        MI_LOG_INFO("[ExecutionContext] Bound tensor for node " + std::to_string(node_id) +
                    " output[" +
                    std::to_string(output_index) + "] to pool " + std::to_string(pool_id) + " (" +
                    std::to_string(required / 1024.0) + " KB, pool size " +
                    std::to_string(pool_size / 1024.0) + " KB)");
    }

    return PoolBindResult::kBound;
}

core::Status ExecutionContext::execute_node(const std::shared_ptr<graph::Node>& node) {
    if (!node) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    if (!node->get_operator()) {
        return core::Status::SUCCESS;
    }

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    const auto& input_edges = node->inputs();
    size_t graph_input_count = 0;
    if (!input_edges.empty()) {
        int max_dst_port = -1;
        for (const auto& edge : input_edges) {
            max_dst_port = std::max(max_dst_port, edge.dst_port);
        }
        graph_input_count = static_cast<size_t>(max_dst_port + 1);
        input_tensors.resize(graph_input_count);
    }

    for (const auto& edge : input_edges) {
        if (!edge.node || edge.dst_port < 0 || edge.src_port < 0) {
            continue;
        }
        if (edge.node->id() >= node_outputs_.size()) {
            continue;
        }
        const auto& src_outputs = node_outputs_[edge.node->id()];
        const size_t src_index = static_cast<size_t>(edge.src_port);
        const size_t dst_index = static_cast<size_t>(edge.dst_port);
        if (src_index >= src_outputs.size() || dst_index >= input_tensors.size()) {
            continue;
        }
        input_tensors[dst_index] = src_outputs[src_index];
    }

    std::vector<std::shared_ptr<core::Tensor>> merged_inputs = node->input_tensors();
    if (merged_inputs.size() < graph_input_count) {
        merged_inputs.resize(graph_input_count);
    }
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        if (input_tensors[i]) {
            merged_inputs[i] = input_tensors[i];
        }
    }

    if (graph_input_count == 0 && merged_inputs.empty()) {
        merged_inputs = input_tensors;
    }

    core::DeviceType device_type = plan_->config().device_type;
    for (const auto& tensor : merged_inputs) {
        if (tensor) {
            device_type = tensor->device();
            break;
        }
    }

    auto context = get_or_create_context(device_type);
    if (!context) {
        MI_LOG_ERROR("[ExecutionContext] No device context for device type");
        return core::Status::ERROR_NOT_IMPLEMENTED;
    }

    auto* previous_context = kernels::get_current_device_context();
    kernels::set_current_device_context(context.get());

    if (node->id() >= node_outputs_.size()) {
        kernels::set_current_device_context(previous_context);
        return core::Status::ERROR_RUNTIME;
    }
    auto& output_tensors = node_outputs_[node->id()];
    auto status = node->get_operator()->forward(merged_inputs, output_tensors);

    kernels::set_current_device_context(previous_context);
    return status;
}

std::shared_ptr<backends::DeviceContext> ExecutionContext::get_or_create_context(
    core::DeviceType device_type) {
    auto it = contexts_.find(device_type);
    if (it != contexts_.end()) {
        return it->second;
    }

    if (device_type == core::DeviceType::CPU) {
        auto context = std::make_shared<backends::CPUDeviceContext>();
        contexts_.emplace(device_type, context);
        return context;
    }

    return nullptr;
}

}  // namespace runtime
}  // namespace mini_infer
