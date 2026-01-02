#include "mini_infer/runtime/execution_context.h"

#include <algorithm>
#include <cstring>

#include "mini_infer/backends/cpu/cpu_allocator.h"
#include "mini_infer/kernels/kernel_registry.h"
#include "mini_infer/runtime/inference_plan.h"
#include "mini_infer/utils/logger.h"

#ifdef MINI_INFER_USE_CUDA
#include "mini_infer/backends/cuda/cuda_device_context.h"
#include "mini_infer/backends/cuda/cuda_allocator.h"
#endif
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

    // Create device context based on configured device type
    core::DeviceType device_type = plan_->config().device_type;
#ifdef MINI_INFER_USE_CUDA
    if (device_type == core::DeviceType::CUDA) {
        contexts_.emplace(core::DeviceType::CUDA,
                          std::make_shared<backends::cuda::CUDADeviceContext>(plan_->config().device_id));
    } else
#endif
    {
        contexts_.emplace(core::DeviceType::CPU, std::make_shared<backends::CPUDeviceContext>());
    }

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

    const bool use_memory_pools = plan_->config().enable_memory_planning;

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
                std::to_string(allocated_count) + " allocated, " + std::to_string(skipped_count) +
                " skipped, " + std::to_string(failed_count) + " failed");

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
        return core::Status::SUCCESS;
    }

    const auto& plan = plan_->get_memory_plan();
    if (plan.shared_buffer_size == 0) {
        MI_LOG_ERROR("[ExecutionContext] Memory planning enabled but shared buffer size is 0");
        return core::Status::ERROR_RUNTIME;
    }

    shared_buffer_size_ = plan.shared_buffer_size;

    // Allocate shared buffer based on device type
    core::DeviceType device_type = plan_->config().device_type;
    void* raw = nullptr;

#ifdef MINI_INFER_USE_CUDA
    if (device_type == core::DeviceType::CUDA) {
        // Use CUDA allocator for GPU memory
        auto cuda_allocator = std::make_shared<backends::cuda::CUDAAllocator>(plan_->config().device_id);
        raw = cuda_allocator->allocate(shared_buffer_size_, plan_->config().memory_alignment);
        if (!raw) {
            MI_LOG_ERROR("[ExecutionContext] Failed to allocate CUDA shared buffer of size " +
                         std::to_string(shared_buffer_size_) + " bytes");
            return core::Status::ERROR_RUNTIME;
        }
        // Zero-initialize CUDA memory
        cudaMemset(raw, 0, shared_buffer_size_);
        // Store allocator to prevent destruction before buffer is freed
        cuda_allocator_ = cuda_allocator;
        shared_buffer_.reset(raw, [allocator = cuda_allocator](void* p) {
            allocator->deallocate(p);
        });
        if (plan_->config().enable_profiling) {
            MI_LOG_INFO("[ExecutionContext] Created CUDA shared buffer (" +
                        std::to_string(shared_buffer_size_ / 1024.0) + " KB)");
        }
    } else
#endif
    {
        // Use CPU allocator for CPU memory
        raw = backends::cpu::CPUAllocator::instance()->allocate(shared_buffer_size_,
                                                                 plan_->config().memory_alignment);
        if (!raw) {
            MI_LOG_ERROR("[ExecutionContext] Failed to allocate shared buffer of size " +
                         std::to_string(shared_buffer_size_) + " bytes");
            return core::Status::ERROR_RUNTIME;
        }
        std::memset(raw, 0, shared_buffer_size_);
        shared_buffer_.reset(raw, [](void* p) { backends::cpu::CPUAllocator::instance()->deallocate(p); });
        if (plan_->config().enable_profiling) {
            MI_LOG_INFO("[ExecutionContext] Created shared buffer (" +
                        std::to_string(shared_buffer_size_ / 1024.0) + " KB)");
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

        const auto bind_result = try_bind_tensor_to_pool(node->id(), i, tensor, use_memory_pools,
                                                         allocated_count, failed_count);
        if (bind_result == PoolBindResult::kBound) {
            continue;
        }
        if (bind_result == PoolBindResult::kFailed) {
            return core::Status::ERROR_RUNTIME;
        }
        if (use_memory_pools) {
            MI_LOG_ERROR("[ExecutionContext] Missing memory plan entry for node " +
                         std::to_string(node->id()) + " output[" + std::to_string(i) + "]");
            return core::Status::ERROR_RUNTIME;
        }

        try {
            *tensor = core::Tensor(shape, tensor->dtype(), plan_->config().device_type);
            allocated_count++;

            if (plan_->config().enable_profiling) {
                MI_LOG_INFO("[ExecutionContext] Allocated tensor for " + node->name() + " output[" +
                            std::to_string(i) + "]: " + shape.to_string() + " (" +
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
                         std::to_string(output_index) + "] requires " + std::to_string(required) +
                         " bytes at offset " + std::to_string(offset) +
                         ", exceeds shared buffer size " + std::to_string(shared_buffer_size_));
            failed_count++;
            return PoolBindResult::kFailed;
        }

        core::DeviceType device_type = plan_->config().device_type;
        if (!tensor->bind_external_data_with_offset(shared_buffer_, shared_buffer_size_, offset,
                                                    device_type)) {
            MI_LOG_ERROR("[ExecutionContext] Failed to bind shared buffer for node " +
                         std::to_string(node_id));
            failed_count++;
            return PoolBindResult::kFailed;
        }

        allocated_count++;
        if (plan_->config().enable_profiling) {
            MI_LOG_INFO("[ExecutionContext] Bound tensor for node " + std::to_string(node_id) +
                        " output[" + std::to_string(output_index) + "] to shared buffer offset " +
                        std::to_string(offset) + " (" + std::to_string(required / 1024.0) + " KB)");
        }
        return PoolBindResult::kBound;
    }

    return PoolBindResult::kNotTried;
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

    // Use the configured device type from the plan
    core::DeviceType device_type = plan_->config().device_type;

#ifdef MINI_INFER_USE_CUDA
    if (device_type == core::DeviceType::CUDA) {
        auto ensure_on_gpu = [&](std::shared_ptr<core::Tensor>& tensor) -> core::Status {
            if (!tensor || tensor->device() == core::DeviceType::CUDA) {
                return core::Status::SUCCESS;
            }

            auto cache_it = gpu_constant_cache_.find(tensor.get());
            if (cache_it != gpu_constant_cache_.end() && cache_it->second) {
                tensor = cache_it->second;
                return core::Status::SUCCESS;
            }

            auto gpu_tensor = std::make_shared<core::Tensor>(
                tensor->shape(), tensor->dtype(), core::DeviceType::CUDA);
            size_t size_bytes = tensor->size_in_bytes();
            cudaError_t status =
                cudaMemcpy(gpu_tensor->data(), tensor->data(), size_bytes, cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                MI_LOG_ERROR("[ExecutionContext] Failed to copy tensor '" + node->name() +
                             "' input to GPU: " + std::string(cudaGetErrorString(status)));
                return core::Status::ERROR_RUNTIME;
            }
            gpu_constant_cache_[tensor.get()] = gpu_tensor;
            tensor = gpu_tensor;
            return core::Status::SUCCESS;
        };

        for (auto& tensor : merged_inputs) {
            auto status = ensure_on_gpu(tensor);
            if (status != core::Status::SUCCESS) {
                return status;
            }
        }
    }
#endif

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

#ifdef MINI_INFER_USE_CUDA
    if (device_type == core::DeviceType::CUDA) {
        auto context = std::make_shared<backends::cuda::CUDADeviceContext>(plan_->config().device_id);
        contexts_.emplace(device_type, context);
        return context;
    }
#endif

    return nullptr;
}

}  // namespace runtime
}  // namespace mini_infer
