# Mini-Infer 快速入门 (Getting Started)

本文档将带你快速上手 Mini-Infer，体验如何加载 ONNX 模型并进行高性能推理。

## 1. 准备工作

### 安装 Conan 并构建项目

```bash
# 1. 安装 Conan
pip install conan

# 2. 检测配置
conan profile detect --force

# 3. 安装依赖并生成预设
conan install . --output-folder=build --build=missing -s build_type=Debug

# 4. 配置并编译
cmake --preset conan-debug
cmake --build --preset conan-debug
```

详细的构建说明请参考 [构建指南](BUILD.md) 或 [快速开始](../QUICK_START.md)。

## 2. 核心概念

在开始写代码前，理解以下三个核心对象：

1.  **Graph**: 静态的计算图拓扑结构（通常由 ONNX Parser 生成）。
2.  **Engine**: 管理 `InferencePlan`（包含优化后的图和内存规划），它是只读且线程安全的。
3.  **ExecutionContext**: 每次推理请求的上下文（包含 Tensor 数据），它是轻量且可变的。

## 3. 完整示例：加载 ONNX 并推理

这个示例展示了标准的推理流程：

```cpp
#include "mini_infer/runtime/engine.h"
#include "mini_infer/importers/onnx_parser.h"
#include "mini_infer/core/tensor.h"
#include <iostream>
#include <vector>

using namespace mini_infer;

int main() {
    // ----------------------------------------------------------------
    // Step 1: 加载模型 (Load Model)
    // ----------------------------------------------------------------
    importers::OnnxParser parser;
    auto graph = parser.parse_from_file("lenet5.onnx");
    
    if (!graph) {
        std::cerr << "Failed to parse ONNX model: " << parser.get_error() << std::endl;
        return -1;
    }
    std::cout << "Successfully parsed graph with " << graph->nodes().size() << " nodes." << std::endl;

    // ----------------------------------------------------------------
    // Step 2: 初始化引擎 (Initialize Engine)
    // ----------------------------------------------------------------
    runtime::EngineConfig config;
    config.device_type = core::DeviceType::CPU;
    config.enable_memory_planning = true;  // 启用静态内存规划 (High Performance)
    config.enable_profiling = false;       // 可选：启用性能分析

    runtime::Engine engine(config);
    
    // Build 阶段会进行：
    // 1. Shape 推导
    // 2. 静态内存规划 (计算每个 Tensor 的 Offset)
    // 3. 准备执行计划
    core::Status status = engine.build(graph);
    if (status != core::Status::SUCCESS) {
        std::cerr << "Engine build failed." << std::endl;
        return -1;
    }

    // ----------------------------------------------------------------
    // Step 3: 准备数据 (Prepare Input)
    // ----------------------------------------------------------------
    // 创建一个 ExecutionContext（这是一个轻量级操作）
    auto ctx = engine.create_context();

    // 假设模型输入名为 "input.1"，形状为 [1, 1, 32, 32]
    auto input_tensor = core::Tensor::create({1, 1, 32, 32}, core::DataType::FLOAT32);
    
    // 填充一些假数据
    float* data = input_tensor->data<float>();
    for (int i = 0; i < 32*32; ++i) data[i] = 1.0f;

    // 将输入绑定到 Context
    ctx->set_input("input.1", input_tensor);

    // ----------------------------------------------------------------
    // Step 4: 执行推理 (Execute)
    // ----------------------------------------------------------------
    // Engine 是无状态的，所有的状态都在 Context 中
    status = engine.execute(ctx.get());
    
    if (status != core::Status::SUCCESS) {
        std::cerr << "Execution failed." << std::endl;
        return -1;
    }

    // ----------------------------------------------------------------
    // Step 5: 获取结果 (Get Output)
    // ----------------------------------------------------------------
    // 假设模型输出名为 "19"
    auto output_tensor = ctx->get_output("19");
    
    if (output_tensor) {
        std::cout << "Inference Output Shape: " << output_tensor->shape().to_string() << std::endl;
        // 打印前几个结果
        const float* out_data = output_tensor->data<float>();
        for (int i = 0; i < 10 && i < output_tensor->shape().numel(); ++i) {
            std::cout << out_data[i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

## 4. 进阶：如何创建计算图 (手动方式)

虽然我们推荐使用 ONNX，但你也可以手动构建图（主要用于测试）：

```cpp
auto graph = std::make_shared<graph::Graph>();

// 1. 创建节点
auto conv = graph->create_node("Conv_1");
// 设置 Operator Param (Conv2DParam) ...

auto relu = graph->create_node("ReLU_1");

// 2. 连接
// 注意：现在支持端口 (src_port, dst_port)
// 将 Conv_1 的第 0 个输出连接到 ReLU_1 的第 0 个输入
graph->connect("Conv_1", "ReLU_1", 0, 0);

// 3. 标记图的 IO
graph->set_inputs({"Conv_1"});
graph->set_outputs({"ReLU_1"});
```

## 5. 常见问题 (FAQ)

**Q: 为什么找不到 `BackendFactory` 了？**
A: 在新架构中，用户不再手动管理 Backend。Engine 会根据 Config 自动在内部管理 `DeviceContext`。如果需要扩展 Backend，请实现新的 `DeviceContext` 类并注册。

**Q: 如何进行多线程推理？**
A: `Engine` 对象是线程安全的。你可以在多个线程中，分别为每个请求创建一个 `ExecutionContext` (ctx1, ctx2...)，然后并发调用 `engine.execute(ctx1)` 和 `engine.execute(ctx2)`。这是处理高并发请求的标准模式。

**Q: 内存是如何管理的？**
A: 启用了 `enable_memory_planning` 后，Engine 会计算出一块足够大的连续内存来容纳所有中间 Tensor。Context 创建时，只会分配这一大块内存（称为 Arena），所有中间 Tensor 实际上都是这就大块内存的 View（通过 Offset 实现）。这极大地提高了缓存命中率并消除了内存碎片。
