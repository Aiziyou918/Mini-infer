# Static Memory Planner - 设计文档

## 概述

实现TensorRT风格的静态内存规划，通过分析Tensor生命周期，让生命周期不重叠的Tensor复用同一块内存，大幅降低内存占用。

---

## 问题分析

### 当前状态（未优化）
```
Layer1_Out: [====生命周期====]     占用 1MB
Layer2_Out:          [====生命周期====]     占用 0.5MB
Layer3_Out:                   [====生命周期====]     占用 0.8MB

总内存占用: 1MB + 0.5MB + 0.8MB = 2.3MB
```

### 优化后（静态内存规划）
```
Memory Pool A: [====Layer1_Out====][====Layer3_Out====]
Memory Pool B:          [====Layer2_Out====]

总内存占用: max(1MB, 0.8MB) + 0.5MB = 1.5MB
节省: 0.8MB (35%)
```

---

## TensorRT 内存规划策略

### 1. **生命周期分析（Liveness Analysis）**
- 确定每个Tensor的生命周期：从创建到最后一次使用
- 使用拓扑排序确定执行顺序
- 标记每个Tensor的 `birth_time` 和 `death_time`

### 2. **内存分配算法**
TensorRT使用**贪心着色算法（Greedy Coloring）**：
- 将Tensor看作图的节点
- 生命周期重叠的Tensor之间有边（冲突）
- 图着色问题：用最少的颜色给节点着色，相邻节点颜色不同
- 每种颜色对应一个内存池

### 3. **内存池管理**
- 每个内存池是一块连续内存
- 池的大小 = 该池中最大Tensor的大小
- Tensor在运行时从对应的池中获取内存（offset=0）

---

## 实现方案

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Planner                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Liveness Analyzer                                       │
│     ├─ Topological Sort                                     │
│     ├─ Compute Birth/Death Time                             │
│     └─ Build Interference Graph                             │
│                                                             │
│  2. Memory Allocator (Greedy Coloring)                      │
│     ├─ Graph Coloring Algorithm                             │
│     ├─ Pool Assignment                                      │
│     └─ Memory Layout Optimization                           │
│                                                             │
│  3. Memory Pool Manager                                     │
│     ├─ Pool Creation                                        │
│     ├─ Tensor Binding                                       │
│     └─ Runtime Memory Access                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 核心数据结构

```cpp
// Tensor生命周期信息
struct TensorLifetime {
    std::string name;
    size_t size_bytes;
    int birth_time;   // 创建时间（拓扑序号）
    int death_time;   // 最后使用时间
    int pool_id;      // 分配的内存池ID
};

// 内存池
struct MemoryPool {
    int pool_id;
    size_t size_bytes;  // 池大小（该池中最大Tensor的大小）
    std::vector<std::string> tensors;  // 使用该池的Tensor列表
};

// 内存规划结果
struct MemoryPlan {
    std::vector<MemoryPool> pools;
    std::unordered_map<std::string, int> tensor_to_pool;
    size_t total_memory;
    size_t original_memory;
    float memory_saving_ratio;
};
```

---

## 算法详解

### 算法1：生命周期分析

```cpp
LivenessAnalyzer::analyze(Graph* graph) {
    // Step 1: 拓扑排序，确定执行顺序
    vector<Node*> topo_order = graph->topological_sort();
    
    // Step 2: 为每个节点分配时间戳
    for (int i = 0; i < topo_order.size(); ++i) {
        node_time[topo_order[i]] = i;
    }
    
    // Step 3: 计算每个Tensor的生命周期
    for (auto& tensor : all_tensors) {
        // Birth time: 生产该Tensor的节点的时间
        tensor.birth_time = node_time[tensor.producer];
        
        // Death time: 最后一个消费该Tensor的节点的时间
        tensor.death_time = 0;
        for (auto& consumer : tensor.consumers) {
            tensor.death_time = max(tensor.death_time, node_time[consumer]);
        }
    }
    
    return lifetimes;
}
```

### 算法2：冲突图构建

```cpp
InterferenceGraph build_interference_graph(vector<TensorLifetime>& lifetimes) {
    InterferenceGraph graph;
    
    // 添加所有Tensor作为节点
    for (auto& lt : lifetimes) {
        graph.add_node(lt.name);
    }
    
    // 添加边：生命周期重叠的Tensor之间有边
    for (int i = 0; i < lifetimes.size(); ++i) {
        for (int j = i + 1; j < lifetimes.size(); ++j) {
            if (lifetimes_overlap(lifetimes[i], lifetimes[j])) {
                graph.add_edge(lifetimes[i].name, lifetimes[j].name);
            }
        }
    }
    
    return graph;
}

bool lifetimes_overlap(TensorLifetime& a, TensorLifetime& b) {
    // 两个区间重叠的条件
    return !(a.death_time < b.birth_time || b.death_time < a.birth_time);
}
```

### 算法3：贪心着色（内存分配）

```cpp
MemoryPlan greedy_coloring(InterferenceGraph& graph, 
                           vector<TensorLifetime>& lifetimes) {
    // 按大小降序排序（大的Tensor优先分配）
    sort(lifetimes.begin(), lifetimes.end(), 
         [](auto& a, auto& b) { return a.size_bytes > b.size_bytes; });
    
    MemoryPlan plan;
    
    for (auto& tensor : lifetimes) {
        // 找到第一个可用的颜色（内存池）
        int pool_id = find_available_pool(tensor, graph, plan);
        
        if (pool_id == -1) {
            // 需要新的内存池
            pool_id = plan.pools.size();
            plan.pools.push_back(MemoryPool{pool_id, tensor.size_bytes, {tensor.name}});
        } else {
            // 使用现有内存池
            plan.pools[pool_id].tensors.push_back(tensor.name);
            plan.pools[pool_id].size_bytes = max(plan.pools[pool_id].size_bytes, 
                                                   tensor.size_bytes);
        }
        
        plan.tensor_to_pool[tensor.name] = pool_id;
    }
    
    return plan;
}

int find_available_pool(TensorLifetime& tensor, 
                        InterferenceGraph& graph,
                        MemoryPlan& plan) {
    for (int pool_id = 0; pool_id < plan.pools.size(); ++pool_id) {
        bool can_use = true;
        
        // 检查该池中的所有Tensor是否与当前Tensor冲突
        for (auto& other_tensor : plan.pools[pool_id].tensors) {
            if (graph.has_edge(tensor.name, other_tensor)) {
                can_use = false;
                break;
            }
        }
        
        if (can_use) {
            return pool_id;
        }
    }
    
    return -1;  // 没有可用的池
}
```

---

## 优化技巧

### 1. **In-place Operations**
某些操作可以原地修改输入（如ReLU），不需要额外内存：
```cpp
if (is_inplace_op(node)) {
    output_tensor.pool_id = input_tensor.pool_id;
    output_tensor.offset = input_tensor.offset;
}
```

### 2. **Persistent Tensors**
某些Tensor需要在整个推理过程中保持（如权重、输入、输出）：
```cpp
if (is_persistent(tensor)) {
    // 分配独立内存，不参与复用
    tensor.pool_id = PERSISTENT_POOL;
}
```

### 3. **Alignment**
内存对齐以提高访问效率：
```cpp
size_t aligned_size = align_up(tensor.size_bytes, 256);  // 256字节对齐
```

---

## 性能预期

### LeNet-5 示例
```
未优化:
  Conv1_out: 6x12x12 = 864 bytes
  Pool1_out: 6x6x6 = 216 bytes
  Conv2_out: 16x4x4 = 256 bytes
  Pool2_out: 16x2x2 = 64 bytes
  FC1_out: 120 bytes
  FC2_out: 84 bytes
  FC3_out: 10 bytes
  总计: ~1.6KB

优化后:
  Pool A: max(864, 256, 120, 10) = 864 bytes
  Pool B: max(216, 64, 84) = 216 bytes
  总计: ~1.1KB
  节省: 31%
```

### 大型网络（如ResNet-50）
- 未优化: ~200MB
- 优化后: ~50MB
- **节省: 75%** 🎉

---

## 实现计划

### Phase 1: 核心框架
- [ ] `MemoryPlanner` 基类
- [ ] `LivenessAnalyzer` 生命周期分析
- [ ] `TensorLifetime` 数据结构

### Phase 2: 内存分配
- [ ] `InterferenceGraph` 冲突图
- [ ] `GreedyColoringAllocator` 贪心着色算法
- [ ] `MemoryPool` 内存池管理

### Phase 3: 集成到Runtime
- [ ] 修改 `Engine::build()` 调用内存规划
- [ ] 修改 `Tensor` 类支持共享内存
- [ ] 运行时内存绑定

### Phase 4: 优化和测试
- [ ] In-place操作优化
- [ ] 内存对齐优化
- [ ] 性能测试和验证

---

## 参考资料

1. **TensorRT Documentation**: Memory Management
2. **TFLite**: Arena Planner
3. **ONNX Runtime**: Memory Pattern Optimization
4. **论文**: "Optimizing Memory Allocation for Deep Neural Networks"

---

## 下一步

开始实现 Phase 1: 核心框架！
