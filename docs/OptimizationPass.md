# Optimization Pass 开发指南

## 概述

Mini-Infer 采用 TensorRT 风格的优化 Pass 注册机制。优化 Pass 会在 Engine build 阶段自动加载和执行，无需修改 Engine 代码。

## 设计理念

参考 TensorRT 的设计：
- **解耦性**: 优化 Pass 独立开发，不影响 Engine 核心逻辑
- **可扩展性**: 通过宏自动注册，添加新 Pass 无需修改现有代码
- **优先级控制**: Pass 按优先级顺序执行
- **统计信息**: 自动收集优化效果统计

## 创建新的优化 Pass

### 1. 定义 Pass 类

```cpp
// include/mini_infer/graph/my_optimization_pass.h
#pragma once

#include "mini_infer/graph/graph_optimizer.h"

namespace mini_infer {
namespace graph {

class MyOptimizationPass : public OptimizationPass {
public:
    MyOptimizationPass();
    ~MyOptimizationPass() override = default;

    core::Status apply(Graph* graph, int& num_modifications) override;

private:
    // Your helper methods
    bool try_optimize_pattern(Graph* graph, ...);
};

} // namespace graph
} // namespace mini_infer
```

### 2. 实现 Pass 逻辑

```cpp
// src/graph/my_optimization_pass.cpp
#include "mini_infer/graph/my_optimization_pass.h"
#include "mini_infer/utils/logger.h"

namespace mini_infer {
namespace graph {

MyOptimizationPass::MyOptimizationPass() 
    : OptimizationPass("MyOptimizationPass") {}

core::Status MyOptimizationPass::apply(Graph* graph, int& num_modifications) {
    if (!graph) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    num_modifications = 0;

    // Your optimization logic here
    for (const auto& [name, node] : graph->nodes()) {
        if (try_optimize_pattern(graph, node)) {
            num_modifications++;
        }
    }

    if (num_modifications > 0) {
        MI_LOG_INFO("[MyOptimizationPass] Applied " + 
                    std::to_string(num_modifications) + " optimization(s)");
    }

    return core::Status::SUCCESS;
}

bool MyOptimizationPass::try_optimize_pattern(Graph* graph, ...) {
    // Pattern matching and transformation logic
    return false;
}

} // namespace graph
} // namespace mini_infer

// 自动注册 Pass (优先级: 数字越小越先执行)
namespace {
std::shared_ptr<mini_infer::graph::OptimizationPass> create_MyOptimizationPass() {
    return std::make_shared<mini_infer::graph::MyOptimizationPass>();
}
struct MyOptimizationPass_Register {
    MyOptimizationPass_Register() {
        mini_infer::graph::OptimizationPassRegistry::instance()
            .register_pass("MyOptimizationPass", create_MyOptimizationPass, 200);
    }
};
static MyOptimizationPass_Register g_MyOptimizationPass_register;
} // namespace
```

### 3. 优先级说明

优化 Pass 按优先级顺序执行（数字越小越先执行）：

| 优先级范围 | 推荐用途 |
|-----------|---------|
| 0-50      | 高优先级优化（如死代码消除） |
| 51-100    | 标准优化（如算子融合） |
| 101-200   | 低优先级优化（如内存优化） |

**当前已注册的 Pass**:
- `FusionPass`: 算子融合 (优先级 100)

## 常见优化模式

### 算子融合 (Operator Fusion)

```cpp
// Conv + ReLU -> Conv(fused_activation=ReLU)
bool try_fuse_conv_activation(Graph* graph, Node* conv_node) {
    // 1. Check pattern
    if (conv_node->outputs().size() != 1) return false;
    auto activation_node = conv_node->outputs()[0];
    if (activation_node->type() != OpType::kRELU) return false;
    
    // 2. Fuse
    auto conv_op = std::static_pointer_cast<Conv2d>(conv_node->get_operator());
    conv_op->set_fused_activation(ActivationType::RELU);
    
    // 3. Reconnect graph
    // ... (bypass activation node)
    
    // 4. Mark for deletion
    graph->remove_node(activation_node->name());
    
    return true;
}
```

### 常量折叠 (Constant Folding)

```cpp
// Add(Constant(2), Constant(3)) -> Constant(5)
bool try_fold_constants(Graph* graph, Node* node) {
    if (node->type() != OpType::kADD) return false;
    
    // Check if all inputs are constants
    for (auto& input : node->inputs()) {
        if (!is_constant(input)) return false;
    }
    
    // Compute result at build time
    auto result = compute_constant_result(node);
    
    // Replace node with constant
    replace_with_constant(graph, node, result);
    
    return true;
}
```

### 死代码消除 (Dead Code Elimination)

```cpp
bool eliminate_dead_nodes(Graph* graph) {
    std::unordered_set<std::string> live_nodes;
    
    // Mark phase: Start from outputs
    for (const auto& output_name : graph->outputs()) {
        mark_live_recursive(graph, output_name, live_nodes);
    }
    
    // Sweep phase: Remove unmarked nodes
    int removed = 0;
    for (const auto& [name, node] : graph->nodes()) {
        if (live_nodes.count(name) == 0) {
            graph->remove_node(name);
            removed++;
        }
    }
    
    return removed > 0;
}
```

## 最佳实践

### 1. 两阶段删除（TensorRT 风格）

```cpp
core::Status apply(Graph* graph, int& num_modifications) {
    std::unordered_set<std::string> nodes_to_delete;
    
    // Phase 1: Mark - 遍历并标记要删除的节点
    for (const auto& [name, node] : graph->nodes()) {
        if (should_remove(node)) {
            nodes_to_delete.insert(name);
        }
    }
    
    // Phase 2: Sweep - 批量删除
    for (const auto& node_name : nodes_to_delete) {
        graph->remove_node(node_name);
    }
    
    return core::Status::SUCCESS;
}
```

### 2. 使用 OpType 快速匹配（避免字符串比较）

```cpp
// Good: 使用 OpType enum
if (node->type() == core::OpType::kCONVOLUTION) {
    // Fast comparison
}

// Bad: 使用字符串
if (node->type_name() == "Convolution") {
    // Slow string comparison
}
```

### 3. 日志记录

```cpp
// 记录优化前后的变化
if (num_modifications > 0) {
    MI_LOG_INFO("[MyPass] Applied " + std::to_string(num_modifications) + 
                " optimization(s)");
}

// 详细调试信息（仅在 verbose 模式）
if (verbose_) {
    MI_LOG_DEBUG("[MyPass] Pattern matched: " + node->name());
}
```

## Engine 配置

优化 Pass 在 Engine build 阶段自动执行：

```cpp
// C++ 代码
mini_infer::runtime::EngineConfig config;
config.enable_graph_optimization = true;  // 启用图优化
config.enable_profiling = true;           // 启用详细日志

mini_infer::runtime::Engine engine(config);
engine.build(graph);

// 获取优化统计
auto stats = engine.get_optimization_stats();
std::cout << "Total modifications: " << stats.total_modifications << std::endl;
```

## 调试技巧

### 1. 查看已注册的 Pass

```cpp
auto& registry = OptimizationPassRegistry::instance();
// 添加一个 get_registered_passes() 方法查看所有 Pass
```

### 2. 禁用特定 Pass（用于调试）

```cpp
GraphOptimizer optimizer;
// 手动添加 Pass，跳过某些 Pass
optimizer.add_pass(std::make_shared<FusionPass>());
// 不添加 MyOptimizationPass
optimizer.optimize(graph);
```

### 3. 单独测试 Pass

```cpp
// 单元测试
TEST(MyOptimizationPassTest, FusesPattern) {
    auto graph = create_test_graph();
    MyOptimizationPass pass;
    int modifications = 0;
    pass.apply(graph.get(), modifications);
    EXPECT_GT(modifications, 0);
}
```

## 参考资料

- TensorRT Optimization Pipeline
- LLVM Pass Infrastructure
- 现有实现: `src/graph/fusion_pass.cpp`

## 示例：添加 Constant Folding Pass

完整示例请参考 `examples/custom_optimization_pass.cpp`（待添加）。

