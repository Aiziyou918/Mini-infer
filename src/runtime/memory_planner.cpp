#include "mini_infer/runtime/memory_planner.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "mini_infer/core/types.h"
#include "mini_infer/utils/logger.h"

namespace mini_infer {
namespace runtime {

// ============================================================================
// InterferenceGraph Implementation
// ============================================================================

void InterferenceGraph::add_node(const std::string& tensor_name) {
    nodes_.insert(tensor_name);
    if (adjacency_list_.find(tensor_name) == adjacency_list_.end()) {
        adjacency_list_[tensor_name] = std::unordered_set<std::string>();
    }
}

void InterferenceGraph::add_edge(const std::string& tensor1, const std::string& tensor2) {
    adjacency_list_[tensor1].insert(tensor2);
    adjacency_list_[tensor2].insert(tensor1);
}

bool InterferenceGraph::has_edge(const std::string& tensor1, const std::string& tensor2) const {
    auto it = adjacency_list_.find(tensor1);
    if (it == adjacency_list_.end()) {
        return false;
    }
    return it->second.count(tensor2) > 0;
}

std::vector<std::string> InterferenceGraph::get_neighbors(const std::string& tensor_name) const {
    auto it = adjacency_list_.find(tensor_name);
    if (it == adjacency_list_.end()) {
        return {};
    }
    return std::vector<std::string>(it->second.begin(), it->second.end());
}

// ============================================================================
// LivenessAnalyzer Implementation
// ============================================================================

std::vector<TensorLifetime> LivenessAnalyzer::analyze(graph::Graph* graph) {
    if (!graph) {
        return {};
    }

    std::vector<TensorLifetime> lifetimes;

    // Step 1: 拓扑排序，确定节点执行顺序
    std::vector<std::shared_ptr<graph::Node>> topo_order;
    auto status = graph->topological_sort(topo_order);
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("[MemoryPlanner] Topological sort failed");
        return {};
    }

    // Step 2: 为每个节点分配时间戳
    std::unordered_map<std::string, int> node_time;
    for (size_t i = 0; i < topo_order.size(); ++i) {
        node_time[topo_order[i]->name()] = static_cast<int>(i);
    }

    // Step 3: 收集所有Tensor及其生产者/消费者
    std::unordered_map<std::string, std::string> producers;  // tensor -> producer_node
    std::unordered_map<std::string, std::vector<std::string>>
        consumers;  // tensor -> consumer_nodes
    compute_producers_consumers(graph, producers, consumers);

    // Step 4: 计算每个Tensor的生命周期
    for (const auto& [tensor_name, producer_name] : producers) {
        TensorLifetime lifetime;
        lifetime.name = tensor_name;

        // Birth time: 生产该Tensor的节点的时间
        auto it = node_time.find(producer_name);
        if (it != node_time.end()) {
            lifetime.birth_time = it->second;
        } else {
            // 可能是图的输入Tensor
            lifetime.birth_time = 0;
        }

        // Death time: 最后一个消费该Tensor的节点的时间
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

        // 检查是否是持久化Tensor
        lifetime.is_persistent = is_persistent_tensor(tensor_name, graph);

        // 计算Tensor大小（如果有shape信息）
        lifetime.size_bytes = 0;
        auto node = graph->get_node(tensor_name);
        if (node && !node->output_tensors().empty() && node->output_tensors()[0]) {
            auto t = node->output_tensors()[0];
            const auto numel = t->shape().numel();
            // 形状未推断时，size_in_bytes 也会为 0，需要兜底
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

    // 遍历所有节点，收集输出Tensor
    for (const auto& [name, node] : graph->nodes()) {
        if (!node)
            continue;

        // 节点的输出Tensor通常以节点名命名
        tensors.push_back(name);
    }

    return tensors;
}

void LivenessAnalyzer::compute_producers_consumers(
    graph::Graph* graph, std::unordered_map<std::string, std::string>& producers,
    std::unordered_map<std::string, std::vector<std::string>>& consumers) {
    // 遍历所有节点
    for (const auto& [node_name, node] : graph->nodes()) {
        if (!node)
            continue;

        // 该节点是其输出Tensor的生产者
        producers[node_name] = node_name;

        // 该节点消费其输入节点的输出
        for (const auto& edge : node->inputs()) {
            if (!edge.node)
                continue;
            consumers[edge.node->name()].push_back(node_name);
        }
    }
}

bool LivenessAnalyzer::is_persistent_tensor(const std::string& tensor_name, graph::Graph* graph) {
    // 检查是否是图的输入或输出
    if (graph->is_input(tensor_name) || graph->is_output(tensor_name)) {
        return true;
    }

    // TODO: 检查是否是权重Tensor（通常权重节点没有输入）
    auto node = graph->get_node(tensor_name);
    if (node && node->inputs().empty()) {
        return true;  // 可能是权重或常量
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

    // Step 1: 生命周期分析
    auto lifetimes = liveness_analyzer_.analyze(graph);
    if (lifetimes.empty()) {
        MI_LOG_WARNING("[MemoryPlanner] No tensors found for memory planning");
        return plan;
    }

    MI_LOG_INFO("[MemoryPlanner] Analyzed " + std::to_string(lifetimes.size()) + " tensors");

    // 计算原始内存占用
    size_t original_memory = 0;
    for (const auto& lt : lifetimes) {
        if (!lt.is_persistent) {
            original_memory += align_size(lt.size_bytes);
        }
    }

    // Step 2: 构建冲突图
    auto interference_graph = build_interference_graph(lifetimes);
    MI_LOG_INFO("[MemoryPlanner] Built interference graph with " +
                std::to_string(interference_graph.nodes().size()) + " nodes");

    // Step 3: 贪心着色算法分配内存池
    plan = allocate_offsets(lifetimes);
    // 恢复原始内存统计，避免被新 plan 覆盖
    plan.original_memory = original_memory;

    // Step 4: 计算统计信息
    plan.compute_statistics();

    // Step 5: 打印结果
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

InterferenceGraph MemoryPlanner::build_interference_graph(
    const std::vector<TensorLifetime>& lifetimes) {
    InterferenceGraph graph;

    // 添加所有非持久化Tensor作为节点
    for (const auto& lt : lifetimes) {
        if (!lt.is_persistent) {
            graph.add_node(lt.name);
        }
    }

    // 添加边：生命周期重叠的Tensor之间有边
    for (size_t i = 0; i < lifetimes.size(); ++i) {
        if (lifetimes[i].is_persistent)
            continue;

        for (size_t j = i + 1; j < lifetimes.size(); ++j) {
            if (lifetimes[j].is_persistent)
                continue;

            if (lifetimes_overlap(lifetimes[i], lifetimes[j])) {
                graph.add_edge(lifetimes[i].name, lifetimes[j].name);
            }
        }
    }

    return graph;
}

bool MemoryPlanner::lifetimes_overlap(const TensorLifetime& a, const TensorLifetime& b) const {
    // 两个区间重叠的条件：NOT (a在b之前结束 OR b在a之前结束)
    return !(a.death_time < b.birth_time || b.death_time < a.birth_time);
}

MemoryPlan MemoryPlanner::greedy_coloring(const InterferenceGraph& graph,
                                          std::vector<TensorLifetime>& lifetimes) {
    MemoryPlan plan;

    // 按大小降序排序（大的Tensor优先分配，减少碎片）
    std::vector<TensorLifetime*> sorted_lifetimes;
    for (auto& lt : lifetimes) {
        if (!lt.is_persistent) {
            sorted_lifetimes.push_back(&lt);
        }
    }

    std::sort(sorted_lifetimes.begin(), sorted_lifetimes.end(),
              [](const TensorLifetime* a, const TensorLifetime* b) {
                  return a->size_bytes > b->size_bytes;
              });

    // 贪心着色
    for (auto* tensor : sorted_lifetimes) {
        // 找到第一个可用的内存池
        int pool_id = find_available_pool(*tensor, graph, plan);

        if (pool_id == -1) {
            // 需要新的内存池
            pool_id = static_cast<int>(plan.pools.size());
            size_t aligned_size = align_size(tensor->size_bytes);
            plan.pools.push_back(MemoryPool(pool_id, aligned_size));
            plan.pools.back().tensors.push_back(tensor->name);
        } else {
            // 使用现有内存池
            plan.pools[pool_id].tensors.push_back(tensor->name);
            // 更新池大小为该池中最大Tensor的大小
            size_t aligned_size = align_size(tensor->size_bytes);
            plan.pools[pool_id].size_bytes = std::max(plan.pools[pool_id].size_bytes, aligned_size);
        }

        plan.tensor_to_pool[tensor->name] = pool_id;
        tensor->pool_id = pool_id;
    }

    return plan;
}

int MemoryPlanner::find_available_pool(const TensorLifetime& tensor, const InterferenceGraph& graph,
                                       const MemoryPlan& plan) const {
    for (size_t pool_id = 0; pool_id < plan.pools.size(); ++pool_id) {
        bool can_use = true;

        // 检查该池中的所有Tensor是否与当前Tensor冲突
        for (const auto& other_tensor : plan.pools[pool_id].tensors) {
            if (graph.has_edge(tensor.name, other_tensor)) {
                can_use = false;
                break;
            }
        }

        if (can_use) {
            return static_cast<int>(pool_id);
        }
    }

    return -1;  // 没有可用的池
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

MemoryPlan MemoryPlanner::allocate_offsets(std::vector<TensorLifetime>& lifetimes) {
    MemoryPlan plan;

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

        plan.tensor_offsets[tensor->name] = offset;
        plan.tensor_to_pool[tensor->name] = 0;
        tensor->pool_id = 0;

        active.push_back(LiveAlloc{tensor->name, tensor->death_time, offset, aligned_size});
    }

    plan.shared_buffer_size = align_size(buffer_size);
    if (plan.shared_buffer_size > 0) {
        plan.pools.clear();
        plan.pools.push_back(MemoryPool(0, plan.shared_buffer_size));
        for (const auto& [name, _] : plan.tensor_offsets) {
            plan.pools[0].tensors.push_back(name);
        }
    }

    return plan;
}

void MemoryPlanner::print_plan(const MemoryPlan& plan) const {
    std::ostringstream oss;

    oss << "\n";
    oss << "╔════════════════════════════════════════════════════════════════════╗\n";
    oss << "║              Static Memory Planning Result                         ║\n";
    oss << "╚════════════════════════════════════════════════════════════════════╝\n";
    oss << "\n";

    oss << "Memory Pools: " << plan.pools.size() << "\n";
    oss << "----------------------------------------\n";

    for (const auto& pool : plan.pools) {
        oss << "Pool " << pool.pool_id << ": " << std::fixed << std::setprecision(2)
            << (pool.size_bytes / 1024.0) << " KB\n";
        oss << "  Tensors (" << pool.tensors.size() << "):\n";
        for (const auto& tensor : pool.tensors) {
            oss << "    - " << tensor << "\n";
        }
        oss << "\n";
    }

    oss << "----------------------------------------\n";
    oss << "Original Memory:  " << std::fixed << std::setprecision(2)
        << (plan.original_memory / 1024.0) << " KB\n";
    oss << "Optimized Memory: " << (plan.total_memory / 1024.0) << " KB\n";
    oss << "Memory Saving:    " << (plan.memory_saving_ratio * 100.0f) << "%\n";
    oss << "╚════════════════════════════════════════════════════════════════════╝\n";

    MI_LOG_INFO(oss.str());
}

}  // namespace runtime
}  // namespace mini_infer
