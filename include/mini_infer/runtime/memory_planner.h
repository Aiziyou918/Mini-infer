#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mini_infer/graph/graph.h"


namespace mini_infer {
namespace runtime {

/**
 * @brief Tensor lifetime
 *
 * Record the lifetime of each tensor (from creation to final usage)
 * Used for static memory planning
 */
struct TensorLifetime {
    std::string name;    // Tensor name
    size_t size_bytes;   // Memory size (bytes)
    int birth_time;      // Birth time (topological order)
    int death_time;      // Death time (topological order)
    int pool_id;         // Pool ID (default -1)
    bool is_persistent;  // Whether the tensor is persistent (weights, inputs, outputs, etc.)

    TensorLifetime()
        : size_bytes(0), birth_time(-1), death_time(-1), pool_id(-1), is_persistent(false) {}
};

/**
 * @brief Memory pool
 *
 * A memory pool corresponds to a continuous memory block, and multiple tensors with non-overlapping lifetimes can reuse the same pool
 */
struct MemoryPool {
    int pool_id;                       // Pool ID
    size_t size_bytes;                 // Pool size (the largest tensor size in this pool)
    std::vector<std::string> tensors;  // List of tensors using this pool

    MemoryPool() : pool_id(-1), size_bytes(0) {}
    MemoryPool(int id, size_t size) : pool_id(id), size_bytes(size) {}
};

/**
 * @brief Memory planning result
 *
 * Contains all memory pool information and Tensor to pool mapping
 */
struct MemoryPlan {
    std::vector<MemoryPool> pools;                        // All memory pools
    std::unordered_map<std::string, int> tensor_to_pool;  // Tensor -> Pool ID mapping
    size_t total_memory;                                  // Total memory usage
    size_t original_memory;                               // Original memory usage before optimization
    float memory_saving_ratio;                            // Memory saving ratio

    MemoryPlan() : total_memory(0), original_memory(0), memory_saving_ratio(0.0f) {}

    /**
     * @brief Compute statistics
     */
    void compute_statistics() {
        total_memory = 0;
        for (const auto& pool : pools) {
            total_memory += pool.size_bytes;
        }

        if (original_memory > 0) {
            memory_saving_ratio = 1.0f - static_cast<float>(total_memory) / original_memory;
        }
    }
};

/**
 * @brief Interference graph
 *
 * Used to represent the lifetime conflict relationship between Tensors
 * If two Tensors have overlapping lifetimes, they have an edge between them
 */
class InterferenceGraph {
   public:
    /**
     * @brief Add node (Tensor)
     */
    void add_node(const std::string& tensor_name);

    /**
     * @brief Add edge (conflict relationship)
     */
    void add_edge(const std::string& tensor1, const std::string& tensor2);

    /**
     * @brief Check if two Tensors conflict
     */
    bool has_edge(const std::string& tensor1, const std::string& tensor2) const;

    /**
     * @brief Get all Tensors that conflict with the specified Tensor
     */
    std::vector<std::string> get_neighbors(const std::string& tensor_name) const;

    /**
     * @brief Get all nodes
     */
    const std::unordered_set<std::string>& nodes() const {
        return nodes_;
    }

   private:
    std::unordered_set<std::string> nodes_;
    std::unordered_map<std::string, std::unordered_set<std::string>> adjacency_list_;
};

/**
 * @brief Liveness analyzer
 *
 * Analyze the lifetime of each Tensor in the graph
 */
class LivenessAnalyzer {
   public:
    /**
     * @brief Analyze the lifetime of all Tensors in the graph
     *
     * @param graph Graph
     * @return TensorLifetime list
     */
    std::vector<TensorLifetime> analyze(graph::Graph* graph);

   private:
    /**
     * @brief Collect all Tensors in the graph
     */
    std::vector<std::string> collect_tensors(graph::Graph* graph);

    /**
     * @brief Compute the producer and consumer of each Tensor
     */
    void compute_producers_consumers(
        graph::Graph* graph, std::unordered_map<std::string, std::string>& producers,
        std::unordered_map<std::string, std::vector<std::string>>& consumers);

    /**
     * @brief Determine if a Tensor is persistent (weights, inputs, outputs)
     */
    bool is_persistent_tensor(const std::string& tensor_name, graph::Graph* graph);
};

/**
 * @brief Static memory planner (TensorRT style)
 *
 * Use greedy coloring algorithm to allocate memory, let Tensor with non-overlapping lifetimes reuse the same memory block
 *
 * Algorithm flow:
 * 1. Lifetime analysis: Determine the lifetime of each Tensor
 * 2. Build conflict graph: Tensor with overlapping lifetimes have edges between them
 * 3. Graph coloring: Use the minimum number of colors to color the nodes, adjacent nodes have different colors
 * 4. Memory allocation: Each color corresponds to a memory pool
 */
class MemoryPlanner {
   public:
    MemoryPlanner();
    ~MemoryPlanner() = default;

    /**
     * @brief Generate memory planning for the graph
     *
     * @param graph Graph
     * @return MemoryPlan
     */
    MemoryPlan plan(graph::Graph* graph);

    /**
     * @brief Set whether to enable memory planning
     */
    void set_enabled(bool enabled) {
        enabled_ = enabled;
    }

    /**
     * @brief Set whether to enable verbose logging
     */
    void set_verbose(bool verbose) {
        verbose_ = verbose;
    }

    /**
     * @brief Set memory alignment size (bytes)
     */
    void set_alignment(size_t alignment) {
        alignment_ = alignment;
    }

   private:
    /**
     * @brief Build conflict graph
     */
    InterferenceGraph build_interference_graph(const std::vector<TensorLifetime>& lifetimes);

    /**
     * @brief Check if two lifetimes overlap
     */
    bool lifetimes_overlap(const TensorLifetime& a, const TensorLifetime& b) const;

    /**
     * @brief Use greedy coloring algorithm to allocate memory pools
     */
    MemoryPlan greedy_coloring(const InterferenceGraph& graph,
                               std::vector<TensorLifetime>& lifetimes);

    /**
     * @brief Find available memory pool for Tensor
     *
     * @return Pool ID, -1 means new pool is needed
     */
    int find_available_pool(const TensorLifetime& tensor, const InterferenceGraph& graph,
                            const MemoryPlan& plan) const;

    /**
     * @brief Align memory size
     */
    size_t align_size(size_t size) const;

    /**
     * @brief Print memory planning result
     */
    void print_plan(const MemoryPlan& plan) const;

   private:
    bool enabled_;      // Whether to enable memory planning
    bool verbose_;      // Whether to print verbose logging
    size_t alignment_;  // Memory alignment size (bytes)

    LivenessAnalyzer liveness_analyzer_;
};

}  // namespace runtime
}  // namespace mini_infer
