#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "mini_infer/backends/device_context.h"
#include "mini_infer/core/tensor.h"
#include "mini_infer/core/allocator.h"
#include "mini_infer/runtime/shape_inference_engine.h"
#include "mini_infer/runtime/inference_plan.h"

namespace mini_infer {
namespace runtime {

/**
 * @brief Execution Context
 *
 * Per-request, mutable runtime state:
 * - Memory pools
 * - Tensor activations
 * - Runtime shape inference state
 */
class ExecutionContext {
   public:
    explicit ExecutionContext(std::shared_ptr<const InferencePlan> plan);
    ~ExecutionContext() = default;

    core::Status set_inputs(
        const std::unordered_map<std::string, std::shared_ptr<core::Tensor>>& inputs);
    core::Status set_inputs(const std::vector<std::shared_ptr<core::Tensor>>& inputs);

    const std::vector<std::shared_ptr<core::Tensor>>& outputs() const {
        return ordered_outputs_;
    }

    const std::unordered_map<std::string, std::shared_ptr<core::Tensor>>& named_outputs() const {
        return named_outputs_;
    }

    const std::unordered_map<std::string, std::shared_ptr<core::Tensor>>& tensor_map() const {
        return tensor_map_;
    }

    void clear_io();

   private:
    friend class InferencePlan;

    core::Status initialize();
    core::Status allocate_tensors();
    core::Status prepare_memory_pools(bool use_memory_pools);
    core::Status allocate_node_outputs(const std::shared_ptr<graph::Node>& node, bool use_pools,
                                       int& allocated_count, int& skipped_count, int& failed_count);

    enum class PoolBindResult { kNotTried, kBound, kFailed };

    PoolBindResult try_bind_tensor_to_pool(size_t node_id, size_t output_index,
                                           std::shared_ptr<core::Tensor>& tensor,
                                           bool use_memory_pools, int& allocated_count,
                                           int& failed_count);

    core::Status execute_node(const std::shared_ptr<graph::Node>& node);

    std::shared_ptr<backends::DeviceContext> get_or_create_context(core::DeviceType device_type);

    std::shared_ptr<const InferencePlan> plan_;
    std::vector<std::shared_ptr<void>> memory_pool_buffers_;
    std::shared_ptr<void> shared_buffer_;
    size_t shared_buffer_size_{0};
    std::shared_ptr<core::Allocator> cuda_allocator_;  // Keep CUDA allocator alive for shared_buffer_
#ifdef MINI_INFER_USE_CUDA
    std::unordered_map<std::shared_ptr<const core::Tensor>,
                       std::shared_ptr<core::Tensor>,
                       TensorPtrHash,
                       TensorPtrEqual> gpu_constant_cache_;
#endif
    std::unordered_map<core::DeviceType, std::shared_ptr<backends::DeviceContext>> contexts_;
    std::unique_ptr<ShapeInferenceEngine> shape_inference_engine_;

    std::vector<std::vector<std::shared_ptr<core::Tensor>>> node_outputs_;
    std::vector<std::shared_ptr<core::Tensor>> ordered_inputs_;
    std::vector<std::shared_ptr<core::Tensor>> ordered_outputs_;
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> named_outputs_;
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> tensor_map_;
    bool initialized_{false};
};

}  // namespace runtime
}  // namespace mini_infer
