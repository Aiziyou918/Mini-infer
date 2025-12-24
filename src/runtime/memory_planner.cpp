#include "mini_infer/runtime/memory_planner.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "mini_infer/core/types.h"
#include "mini_infer/utils/logger.h"

namespace mini_infer {
namespace runtime {



// ============================================================================
// LivenessAnalyzer Implementation
// ============================================================================

std::vector<TensorLifetime> LivenessAnalyzer::analyze(graph::Graph* graph) {
    if (!graph) {
        return {};
    }

    std::vector<TensorLifetime> lifetimes;

    // Step 1: Topological sort, determine node execution order
    std::vector<std::shared_ptr<graph::Node>> topo_order;
    auto status = graph->topological_sort(topo_order);
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("[MemoryPlanner] Topological sort failed");
        return {};
    }

    // Step 2: Assign timestamps to each node
    std::unordered_map<std::string, int> node_time;
    for (size_t i = 0; i < topo_order.size(); ++i) {
        node_time[topo_order[i]->name()] = static_cast<int>(i);
    }

    // Step 3: Collect all tensors and their producers/consumers
    std::unordered_map<std::string, std::string> producers;  // tensor -> producer_node
    std::unordered_map<std::string, std::vector<std::string>>
        consumers;  // tensor -> consumer_nodes
    compute_producers_consumers(graph, producers, consumers);

    // Step 4: Calculate the lifetime of each tensor
    for (const auto& [tensor_name, producer_name] : producers) {
        TensorLifetime lifetime;
        lifetime.name = tensor_name;

        // Birth time: The time when the node that produces this tensor
        auto it = node_time.find(producer_name);
        if (it != node_time.end()) {
            lifetime.birth_time = it->second;
        } else {
            // May be graph input tensor
            lifetime.birth_time = 0;
        }

        // Death time: The time when the last node that consumes this tensor
        lifetime.death_time = lifetime.birth_time;
        auto consumer_it = consumers.find(tensor_name);
        if (consumer_it != consumers.end()) {
            for (const auto& consumer_name : consumer_it->second) {
                auto time_it = node_time.find(consumer_name);
                if (time_it != node_time.end()) {
                    lifetime.death_time = std::max(lifetime.death_time, time_it->second);
                }
            }
        }

        // Check if the tensor is persistent
        lifetime.is_persistent = is_persistent_tensor(tensor_name, graph);

        // Calculate tensor size (if shape information is available)
        lifetime.size_bytes = 0;
        auto node = graph->get_node(tensor_name);
        if (node && !node->output_tensors().empty() && node->output_tensors()[0]) {
            lifetime.node_id = node->id();
            auto t = node->output_tensors()[0];
            const auto numel = t->shape().numel();
            // Fallback to 1024 bytes if size_in_bytes is not available
            const size_t bytes = t->size_in_bytes();
            lifetime.size_bytes = bytes > 0 ? bytes
                                            : (numel > 0 ? static_cast<size_t>(numel) * sizeof(float)
                                                         : 1024);
        } else {
            lifetime.size_bytes = 1024;  // fallback
        }

        lifetimes.push_back(lifetime);
    }

    return lifetimes;
}

std::vector<std::string> LivenessAnalyzer::collect_tensors(graph::Graph* graph) {
    std::vector<std::string> tensors;

    // Iterate over all nodes and collect output tensors
    for (const auto& node : graph->nodes()) {
        if (!node) {
            continue;
        }
        // The output tensor of a node is usually named after the node
        tensors.push_back(node->name());
    }

    return tensors;
}

void LivenessAnalyzer::compute_producers_consumers(
    graph::Graph* graph, std::unordered_map<std::string, std::string>& producers,
    std::unordered_map<std::string, std::vector<std::string>>& consumers) {
    // Iterate over all nodes
    for (const auto& node : graph->nodes()) {
        if (!node) {
            continue;
        }

        const std::string& node_name = node->name();
        // The node is the producer of its output tensor
        producers[node_name] = node_name;

        // The node consumes the output of its input nodes
        for (const auto& edge : node->inputs()) {
            if (!edge.node) {
                continue;
            }
            consumers[edge.node->name()].push_back(node_name);
        }
    }
}

bool LivenessAnalyzer::is_persistent_tensor(const std::string& tensor_name, graph::Graph* graph) {
    // Check if the tensor is an input or output
    if (graph->is_input(tensor_name)) {
        return true;
    }

    // TODO: Check if the tensor is a weight tensor (usually weight nodes have no inputs)
    auto node = graph->get_node(tensor_name);
    if (node && node->inputs().empty()) {
        return true; 
    }

    return false;
}

// ============================================================================
// MemoryPlanner Implementation
// ============================================================================

MemoryPlanner::MemoryPlanner() : enabled_(true), verbose_(false), alignment_(256) {}

MemoryPlan MemoryPlanner::plan(graph::Graph* graph) {
    MemoryPlan plan;

    if (!enabled_ || !graph) {
        return plan;
    }

    MI_LOG_INFO("[MemoryPlanner] Starting static memory planning...");

    // Step 1: Lifetime analysis
    auto lifetimes = liveness_analyzer_.analyze(graph);
    if (lifetimes.empty()) {
        MI_LOG_WARNING("[MemoryPlanner] No tensors found for memory planning");
        return plan;
    }

    MI_LOG_INFO("[MemoryPlanner] Analyzed " + std::to_string(lifetimes.size()) + " tensors");

    // Step 2: Calculate original memory usage
    size_t original_memory = 0;
    for (const auto& lt : lifetimes) {
        if (!lt.is_persistent) {
            original_memory += align_size(lt.size_bytes);
        }
    }

    // Step 3: Allocate offsets using linear-scan algorithm
    plan = allocate_offsets(lifetimes, graph->node_capacity());
    plan.original_memory = original_memory;

    // Step 4: Calculate statistics
    plan.compute_statistics();

    // Step 5: Print plan
    if (verbose_) {
        print_plan(plan);
    }

    MI_LOG_INFO("[MemoryPlanner] Memory planning completed");
    MI_LOG_INFO("[MemoryPlanner] Original memory: " +
                std::to_string(plan.original_memory / 1024.0) + " KB");
    MI_LOG_INFO("[MemoryPlanner] Optimized memory: " + std::to_string(plan.total_memory / 1024.0) +
                " KB");
    MI_LOG_INFO("[MemoryPlanner] Memory saving: " +
                std::to_string(plan.memory_saving_ratio * 100.0f) + "%");

    return plan;
}



size_t MemoryPlanner::align_size(size_t size) const {
    if (alignment_ == 0) {
        return size;
    }
    return ((size + alignment_ - 1) / alignment_) * alignment_;
}

size_t MemoryPlanner::align_offset(size_t offset) const {
    if (alignment_ == 0) {
        return offset;
    }
    return ((offset + alignment_ - 1) / alignment_) * alignment_;
}

MemoryPlan MemoryPlanner::allocate_offsets(std::vector<TensorLifetime>& lifetimes,
                                           size_t node_capacity) {
    MemoryPlan plan;
    plan.tensor_offsets.assign(node_capacity, MemoryPlan::kInvalidOffset);
    plan.tensor_to_pool.assign(node_capacity, MemoryPlan::kInvalidPool);

    struct LiveAlloc {
        std::string name;
        int death_time;
        size_t offset;
        size_t size;
    };

    struct FreeBlock {
        size_t offset;
        size_t size;
    };

    std::vector<TensorLifetime*> sorted_lifetimes;
    for (auto& lt : lifetimes) {
        if (!lt.is_persistent) {
            sorted_lifetimes.push_back(&lt);
        }
    }

    std::sort(sorted_lifetimes.begin(), sorted_lifetimes.end(),
              [](const TensorLifetime* a, const TensorLifetime* b) {
                  if (a->birth_time == b->birth_time) {
                      return a->size_bytes > b->size_bytes;
                  }
                  return a->birth_time < b->birth_time;
              });

    std::vector<LiveAlloc> active;
    std::vector<FreeBlock> free_blocks;
    size_t buffer_size = 0;

    auto collect_free_block = [&](size_t offset, size_t size) {
        if (size == 0) {
            return;
        }
        free_blocks.push_back(FreeBlock{offset, size});
        std::sort(free_blocks.begin(), free_blocks.end(),
                  [](const FreeBlock& a, const FreeBlock& b) { return a.offset < b.offset; });
        std::vector<FreeBlock> merged;
        for (const auto& block : free_blocks) {
            if (merged.empty() ||
                merged.back().offset + merged.back().size < block.offset) {
                merged.push_back(block);
            } else {
                merged.back().size =
                    std::max(merged.back().offset + merged.back().size,
                             block.offset + block.size) -
                    merged.back().offset;
            }
        }
        free_blocks.swap(merged);
    };

    for (auto* tensor : sorted_lifetimes) {
        if (tensor->node_id == TensorLifetime::kInvalidNodeId ||
            tensor->node_id >= node_capacity) {
            continue;
        }
        const size_t aligned_size = align_size(tensor->size_bytes);

        // Expire allocations whose lifetime ended before this tensor starts.
        auto it = active.begin();
        while (it != active.end()) {
            if (it->death_time < tensor->birth_time) {
                collect_free_block(it->offset, it->size);
                it = active.erase(it);
            } else {
                ++it;
            }
        }

        size_t offset = 0;
        bool found = false;
        for (auto block_it = free_blocks.begin(); block_it != free_blocks.end(); ++block_it) {
            if (block_it->size >= aligned_size) {
                offset = block_it->offset;
                if (block_it->size == aligned_size) {
                    free_blocks.erase(block_it);
                } else {
                    block_it->offset += aligned_size;
                    block_it->size -= aligned_size;
                }
                found = true;
                break;
            }
        }

        if (!found) {
            offset = align_offset(buffer_size);
            buffer_size = offset + aligned_size;
        }

        plan.tensor_offsets[tensor->node_id] = offset;
        plan.tensor_to_pool[tensor->node_id] = 0;
        tensor->pool_id = 0;

        active.push_back(LiveAlloc{tensor->name, tensor->death_time, offset, aligned_size});
    }

    plan.shared_buffer_size = align_size(buffer_size);
    if (plan.shared_buffer_size > 0) {
        plan.pools.clear();
        plan.pools.push_back(MemoryPool(0, plan.shared_buffer_size));
    }

    return plan;
}

void MemoryPlanner::print_plan(const MemoryPlan& plan) const {
    std::ostringstream oss;

    oss << "\n";
    oss << "Static Memory Planning Result\n";
    oss << "------------------------------\n";
    oss << "Shared Buffer:   " << std::fixed << std::setprecision(2)
        << (plan.shared_buffer_size / 1024.0) << " KB\n";

    size_t allocated = 0;
    for (const auto offset : plan.tensor_offsets) {
        if (offset != MemoryPlan::kInvalidOffset) {
            allocated++;
        }
    }
    oss << "Tensors Allocated: " << allocated << "\n";
    oss << "Original Memory:  " << std::fixed << std::setprecision(2)
        << (plan.original_memory / 1024.0) << " KB\n";
    oss << "Optimized Memory: " << std::fixed << std::setprecision(2)
        << (plan.total_memory / 1024.0) << " KB\n";
    oss << "Memory Saving:    " << std::fixed << std::setprecision(2)
        << (plan.memory_saving_ratio * 100.0f) << "%\n";

    MI_LOG_INFO(oss.str());
}


}  // namespace runtime
}  // namespace mini_infer
