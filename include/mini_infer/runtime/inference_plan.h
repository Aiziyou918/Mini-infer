#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mini_infer/backends/device_context.h"
#include "mini_infer/core/allocator.h"
#include "mini_infer/core/tensor.h"
#include "mini_infer/core/types.h"
#include "mini_infer/graph/graph.h"
#include "mini_infer/graph/graph_optimizer.h"
#include "mini_infer/runtime/memory_planner.h"
#include "mini_infer/runtime/optimization_profile.h"
#include "mini_infer/runtime/shape_inference_engine.h"

namespace mini_infer {
namespace runtime {

class ExecutionContext;

/**
 * @brief Hash function for shared_ptr<const core::Tensor>
 */
struct TensorPtrHash {
    size_t operator()(const std::shared_ptr<const core::Tensor>& ptr) const {
        return std::hash<const core::Tensor*>{}(ptr.get());
    }
};

/**
 * @brief Equality comparator for shared_ptr<const core::Tensor>
 */
struct TensorPtrEqual {
    bool operator()(const std::shared_ptr<const core::Tensor>& lhs,
                    const std::shared_ptr<const core::Tensor>& rhs) const {
        return lhs.get() == rhs.get();
    }
};

/**
 * @brief Config Inference Plan (TensorRT-style)
 */
struct EngineConfig {
    core::DeviceType device_type{core::DeviceType::CPU};
    int32_t device_id{0};
    bool enable_profiling{false};
    bool enable_graph_optimization{true};           // Enable graph optimization
    bool enable_memory_planning{true};              // Enable memory planning
    size_t memory_alignment{256};                   // Memory alignment (bytes)
    size_t max_workspace_size{1024 * 1024 * 1024};  // 1GB

    // Dynamic shape support
    bool enable_dynamic_shapes{false};                          // Enable dynamic shape support
    std::shared_ptr<OptimizationProfile> optimization_profile;  // Shape ranges for inputs
};

/**
 * @brief Inference Plan
 *
 * Owns immutable, thread-safe build artifacts:
 * - Graph
 * - Weights
 * - Optimization profile
 * - Memory plan
 */
class InferencePlan : public std::enable_shared_from_this<InferencePlan> {
   public:
    explicit InferencePlan(const EngineConfig& config);
    ~InferencePlan() = default;

    struct InputBinding {
        std::string name;
        size_t node_id{0};
        graph::Node* node{nullptr};
    };

    /**
     * @brief Build the plan
     */
    core::Status build(std::shared_ptr<graph::Graph> graph);

    /**
     * @brief Create a per-request execution context
     */
    std::shared_ptr<ExecutionContext> create_execution_context() const;

    /**
     * @brief Execute the plan with a prepared context
     */
    core::Status execute(ExecutionContext* ctx) const;

    std::vector<std::string> get_input_names() const;
    std::vector<std::string> get_output_names() const;

    const MemoryPlan& get_memory_plan() const {
        return memory_plan_;
    }

    const graph::GraphOptimizer::Statistics& get_optimization_stats() const {
        return optimization_stats_;
    }

    const EngineConfig& config() const {
        return config_;
    }

    const std::shared_ptr<graph::Graph>& graph() const {
        return graph_;
    }

    const std::vector<std::shared_ptr<graph::Node>>& sorted_nodes() const {
        return sorted_nodes_;
    }

    const std::vector<InputBinding>& input_bindings() const {
        return input_bindings_;
    }

    void set_weights(
        std::unordered_map<std::string, std::shared_ptr<core::Tensor>> weights) {
        weights_ = std::move(weights);
    }

    const std::unordered_map<std::string, std::shared_ptr<core::Tensor>>& weights() const {
        return weights_;
    }

    /**
     * @brief Get GPU tensor for a CPU tensor (TensorRT-style preloaded weights)
     * @param cpu_tensor The original CPU tensor
     * @return GPU tensor if available, nullptr otherwise
     */
    std::shared_ptr<core::Tensor> get_gpu_tensor(
        const std::shared_ptr<const core::Tensor>& cpu_tensor) const;

   private:
    friend class ExecutionContext;

    EngineConfig config_;
    std::shared_ptr<graph::Graph> graph_;
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> weights_;
    std::vector<std::shared_ptr<graph::Node>> sorted_nodes_;
    MemoryPlan memory_plan_;
    graph::GraphOptimizer::Statistics optimization_stats_;
    std::vector<InputBinding> input_bindings_;

    // TensorRT-style: GPU weights preloaded at build time
    std::unordered_map<std::shared_ptr<const core::Tensor>,
                       std::shared_ptr<core::Tensor>,
                       TensorPtrHash,
                       TensorPtrEqual> gpu_weight_cache_;
    std::shared_ptr<core::Allocator> cuda_allocator_;  // Keep CUDA allocator alive

    core::Status optimize_graph();
    core::Status infer_shapes();
    core::Status infer_shapes_with_profile();
    core::Status update_tensor_properties();
    core::Status plan_memory();
    core::Status initialize_input_bindings();
    core::Status preload_weights_to_gpu();  // TensorRT-style weight preloading

    core::Status gather_map_inputs(
        const std::unordered_map<std::string, std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& ordered_inputs) const;
    core::Status build_runtime_shapes(
        const std::vector<std::shared_ptr<core::Tensor>>& ordered_inputs,
        std::vector<ShapeInferenceEngine::RuntimeInputShape>& runtime_shapes) const;
    core::Status bind_ordered_inputs(
        ExecutionContext* ctx,
        const std::vector<std::shared_ptr<core::Tensor>>& ordered_inputs) const;
    core::Status collect_ordered_outputs(ExecutionContext* ctx) const;
    bool check_shape_change(ExecutionContext* ctx,
                            const std::vector<ShapeInferenceEngine::RuntimeInputShape>&
                                runtime_shapes) const;
    core::Status handle_shape_change(
        ExecutionContext* ctx,
        const std::vector<ShapeInferenceEngine::RuntimeInputShape>& runtime_shapes) const;
};

}  // namespace runtime
}  // namespace mini_infer
