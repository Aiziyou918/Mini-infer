#pragma once

#include "mini_infer/graph/graph.h"
#include "mini_infer/graph/graph_optimizer.h"
#include "mini_infer/runtime/memory_planner.h"
#include "mini_infer/backends/backend.h"
#include "mini_infer/core/types.h"
#include <memory>
#include <vector>
#include <unordered_map>

namespace mini_infer {
namespace runtime {

/**
 * @brief Config Inference Engine (TensorRT-style)
 */
struct EngineConfig {
    core::DeviceType device_type{core::DeviceType::CPU};
    int32_t device_id{0};
    bool enable_profiling{false};
    bool enable_graph_optimization{true};     // Enable graph optimization
    bool enable_memory_planning{true};        // Enable memory planning
    size_t memory_alignment{256};             // Memory alignment (bytes)
    size_t max_workspace_size{1024 * 1024 * 1024}; // 1GB
};

/**
 * @brief Inference Engine
 * Execute the graph and manage the inference process
 */
class Engine {
public:
    explicit Engine(const EngineConfig& config);
    ~Engine() = default;
    
    /**
     * @brief Build the engine
     * 
     * @param graph 
     * @return core::Status 
     */
    core::Status build(std::shared_ptr<graph::Graph> graph);
    
    /**
     * @brief Execute the graph
     * 
     * @param inputs input tensors
     * @param outputs output tensors
     * @return core::Status 
     */
    core::Status forward(
        const std::unordered_map<std::string, std::shared_ptr<core::Tensor>>& inputs,
        std::unordered_map<std::string, std::shared_ptr<core::Tensor>>& outputs
    );
    
    /**
     * @brief Get input names
     * 
     * @return std::vector<std::string> 
     */
    std::vector<std::string> get_input_names() const;
    
    /**
     * @brief Get output names
     * 
     * @return std::vector<std::string> 
     */
    std::vector<std::string> get_output_names() const;
    
    /**
     * @brief Enable profiling
     * 
     * @param enable 
     */
    void enable_profiling(bool enable) { config_.enable_profiling = enable; }

    /**
     * @brief Get profiling info
     * 
     * @return std::string 
     */
    std::string get_profiling_info() const;

    /**
     * @brief Get memory plan (if memory planning was enabled)
     */
    const MemoryPlan& get_memory_plan() const { return memory_plan_; }

    /**
     * @brief Get optimization statistics
     */
    const graph::GraphOptimizer::Statistics& get_optimization_stats() const { 
        return optimization_stats_; 
    }
    
private:
    EngineConfig config_; ///< Engine config
    std::shared_ptr<graph::Graph> graph_; ///< Graph
    std::shared_ptr<backends::Backend> backend_; ///< Backend
    std::vector<std::shared_ptr<graph::Node>> sorted_nodes_; ///< Sorted nodes
    MemoryPlan memory_plan_; ///< Memory plan result
    graph::GraphOptimizer::Statistics optimization_stats_; ///< Optimization statistics
    
    /**
     * @brief Apply graph optimizations (TensorRT-style)
     * 
     * @return core::Status 
     */
    core::Status optimize_graph();

    /**
     * @brief Infer shapes for all tensors in the graph
     * 
     * @return core::Status 
     */
    core::Status infer_shapes();

    /**
     * @brief Plan memory allocation (TensorRT-style)
     * 
     * @return core::Status 
     */
    core::Status plan_memory();

    /**
     * @brief Allocate tensors based on shapes and memory plan
     * 
     * @return core::Status 
     */
    core::Status allocate_tensors();
    
    /**
     * @brief Execute node
     * 
     * @param node node will be executed
     * @return core::Status 
     */
    core::Status execute_node(std::shared_ptr<graph::Node> node);
};

} // namespace runtime
} // namespace mini_infer

