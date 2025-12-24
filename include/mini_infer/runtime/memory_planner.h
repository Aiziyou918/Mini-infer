#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
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
    size_t node_id;      // Node ID
    size_t size_bytes;   // Memory size (bytes)
    int birth_time;      // Birth time (topological order)
    int death_time;      // Death time (topological order)
    int pool_id;         // Pool ID (default -1)
    bool is_persistent;  // Whether the tensor is persistent (weights, inputs, outputs, etc.)

    TensorLifetime()
        : node_id(kInvalidNodeId),
          size_bytes(0),
          birth_time(-1),
          death_time(-1),
          pool_id(-1),
          is_persistent(false) {}

    static constexpr size_t kInvalidNodeId = std::numeric_limits<size_t>::max();
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
    std::vector<int> tensor_to_pool;                      // Node ID -> Pool ID mapping
    std::vector<size_t> tensor_offsets;                   // Node ID -> Offset mapping
    size_t shared_buffer_size{0};                            // Shared buffer size in bytes
    size_t total_memory;                                  // Total memory usage
    size_t original_memory;                               // Original memory usage before optimization
    float memory_saving_ratio;                            // Memory saving ratio

    MemoryPlan() : total_memory(0), original_memory(0), memory_saving_ratio(0.0f) {}

    static constexpr size_t kInvalidOffset = std::numeric_limits<size_t>::max();
    static constexpr int kInvalidPool = -1;

    /**
     * @brief Compute statistics
     */
    void compute_statistics() {
        if (shared_buffer_size > 0) {
            total_memory = shared_buffer_size;
        } else {
            total_memory = 0;
            for (const auto& pool : pools) {
                total_memory += pool.size_bytes;
            }
        }

        if (original_memory > 0) {
            memory_saving_ratio = 1.0f - static_cast<float>(total_memory) / original_memory;
        }
    }
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
 * Uses linear-scan algorithm to allocate memory offsets in a single shared buffer.
 * Tensors with non-overlapping lifetimes reuse the same memory regions.
 *
 * Algorithm flow:
 * 1. Lifetime analysis: Determine birth/death time of each tensor
 * 2. Linear scan: Allocate offsets by reusing freed memory blocks
 * 3. Result: Single contiguous buffer with optimal memory reuse
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
     * @brief Allocate offsets in a shared buffer using linear-scan algorithm
     */
    MemoryPlan allocate_offsets(std::vector<TensorLifetime>& lifetimes, size_t node_capacity);

    /**
     * @brief Align memory size
     */
    size_t align_size(size_t size) const;

    size_t align_offset(size_t offset) const;

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
