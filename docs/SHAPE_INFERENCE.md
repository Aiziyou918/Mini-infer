# Shape 推断系统文档

## 概述

Mini-Infer 实现了 TensorRT 风格的自动 Shape 推断系统。在 Engine build 阶段，框架会自动推断所有 Tensor 的形状，无需手动指定。

## 设计理念

### TensorRT 风格的三阶段处理

```cpp
Engine::build(graph) {
    1. Graph Optimization    // 算子融合等优化
    2. Shape Inference       // ⭐ 自动推断所有形状
    3. Memory Planning       // 基于形状分配内存
    4. Tensor Allocation     // 实际分配内存
}
```

### 关键特性

- ✅ **自动推断**：按拓扑序逐层推断，无需手动配置
- ✅ **ONNX 集成**：从 ONNX 模型中解析初始形状
- ✅ **动态维度**：支持动态 batch size（维度值为 -1）
- ✅ **错误检测**：形状不匹配时提前报错
- ✅ **详细日志**：`enable_profiling=true` 时显示每个节点的形状

---

## 工作流程

### 1. ONNX 导入阶段

从 ONNX 模型的 `ValueInfoProto` 中解析输入/输出形状：

```cpp
// src/importers/model_importer.cpp
core::Status ModelImporter::import_inputs(...) {
    for (const auto& input : graph_proto.input()) {
        // 解析 shape
        if (input.has_type() && input.type().has_tensor_type()) {
            const auto& tensor_type = input.type().tensor_type();
            if (tensor_type.has_shape()) {
                std::vector<int64_t> dims;
                for (const auto& dim : tensor_type.shape().dim()) {
                    if (dim.has_dim_value()) {
                        dims.push_back(dim.dim_value());
                    } else {
                        dims.push_back(-1);  // Dynamic dimension
                    }
                }
                input_tensor->reshape(core::Shape(dims));
            }
        }
    }
}
```

**示例：**
```
ONNX Input: [N, 3, 224, 224]  其中 N 是动态维度
解析结果: [-1, 3, 224, 224]
实例化时: [1, 3, 224, 224]  (默认 batch=1)
```

### 2. Engine Build 阶段

Engine 在 `infer_shapes()` 中按拓扑序推断：

```cpp
core::Status Engine::infer_shapes() {
    // 按拓扑序遍历节点
    for (auto& node : sorted_nodes_) {
        // 收集输入形状（从图连接和权重）
        std::vector<core::Shape> input_shapes;
        
        // 1. 从图连接的输入节点获取形状
        for (const auto& input_node : node->inputs()) {
            input_shapes.push_back(input_node->output_tensors()[0]->shape());
        }
        
        // 2. 从导入的权重获取形状
        for (const auto& weight : node->input_tensors()) {
            input_shapes.push_back(weight->shape());
        }
        
        // 3. 调用算子的 infer_shape() 方法
        std::vector<core::Shape> output_shapes;
        node->get_operator()->infer_shape(input_shapes, output_shapes);
        
        // 4. 更新输出 tensor 的形状
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            node->output_tensors()[i]->reshape(output_shapes[i]);
        }
    }
}
```

**示例流程：**
```
Input: [1, 3, 224, 224]
  ↓
Conv2D: infer_shape([1,3,224,224], [64,3,7,7]) → [1, 64, 112, 112]
  ↓
ReLU: infer_shape([1,64,112,112]) → [1, 64, 112, 112]
  ↓
MaxPool: infer_shape([1,64,112,112]) → [1, 64, 56, 56]
  ↓
Flatten: infer_shape([1,64,56,56]) → [1, 200704]
  ↓
Linear: infer_shape([1,200704], [1000,200704]) → [1, 1000]
```

### 3. 算子实现

每个算子实现 `infer_shape()` 方法：

#### Conv2D 示例

```cpp
core::Status Conv2D::infer_shape(
    const std::vector<core::Shape>& input_shapes,
    std::vector<core::Shape>& output_shapes) {
    
    const auto& input_shape = input_shapes[0];   // [N, C_in, H_in, W_in]
    const auto& weight_shape = input_shapes[1];  // [C_out, C_in, kh, kw]
    
    int64_t N = input_shape[0];
    int64_t C_in = input_shape[1];
    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];
    
    int64_t C_out = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];
    
    // 计算输出尺寸
    int64_t H_out = (H_in + 2*padding_h - dilation_h*(kernel_h-1) - 1) / stride_h + 1;
    int64_t W_out = (W_in + 2*padding_w - dilation_w*(kernel_w-1) - 1) / stride_w + 1;
    
    output_shapes = {core::Shape({N, C_out, H_out, W_out})};
    return core::Status::SUCCESS;
}
```

#### Pooling 示例

```cpp
core::Status Pooling::infer_shape(...) {
    const auto& input_shape = input_shapes[0];  // [N, C, H, W]
    
    int64_t H_out = (H_in + 2*padding_h - kernel_h) / stride_h + 1;
    int64_t W_out = (W_in + 2*padding_w - kernel_w) / stride_w + 1;
    
    output_shapes = {core::Shape({N, C, H_out, W_out})};
    return core::Status::SUCCESS;
}
```

#### ReLU 示例

```cpp
core::Status ReLU::infer_shape(...) {
    // ReLU 不改变形状
    output_shapes = {input_shapes[0]};
    return core::Status::SUCCESS;
}
```

---

## 使用方法

### 方法 1：ONNX 模型（推荐）

ONNX 模型自带形状信息，自动推断：

```cpp
#include "mini_infer/importers/onnx_parser.h"
#include "mini_infer/runtime/engine.h"

// 1. 加载 ONNX 模型
mini_infer::importers::OnnxParser parser;
auto graph = parser.parse_from_file("model.onnx");

// 2. 构建 Engine（自动推断形状）
mini_infer::runtime::EngineConfig config;
config.enable_profiling = true;  // 查看详细日志

mini_infer::runtime::Engine engine(config);
engine.build(graph);  // 自动完成形状推断

// 3. 运行推理
auto input = std::make_shared<core::Tensor>(
    core::Shape({1, 3, 224, 224}), 
    core::DataType::FLOAT32
);
outputs = engine.forward({{"input", input}});
```

**日志输出示例：**
```
[Engine] Step 3: Inferring tensor shapes...
[Engine] Node conv1 output[0] shape: [1, 64, 112, 112]
[Engine] Node relu1 output[0] shape: [1, 64, 112, 112]
[Engine] Node pool1 output[0] shape: [1, 64, 56, 56]
[Engine] Shape inference completed: 15 tensor(s) inferred, 0 failed
```

### 方法 2：手动构建图

手动构建时需要设置输入形状：

```cpp
auto graph = std::make_shared<graph::Graph>();

// 创建输入节点并设置形状
auto input_node = graph->create_node("input");
auto input_tensor = std::make_shared<core::Tensor>();
input_tensor->reshape(core::Shape({1, 3, 224, 224}));
input_node->set_output_tensors({input_tensor});

// 创建 Conv2D 节点（权重需要设置形状）
auto conv_node = graph->create_node("conv");
auto conv_op = std::make_shared<operators::Conv2D>(conv_param);
conv_node->set_operator(conv_op);

auto weight = std::make_shared<core::Tensor>(
    core::Shape({64, 3, 7, 7}),  // [out_ch, in_ch, kh, kw]
    core::DataType::FLOAT32
);
conv_node->set_input_tensors({nullptr, weight, bias});  // data 来自图连接

// 连接节点
graph->connect("input", "conv");

// 其余节点的形状会自动推断
```

---

## 动态形状支持

### 动态 Batch Size

```cpp
// ONNX 中定义动态维度
// input shape: [-1, 3, 224, 224]

// 推理时提供实际 batch size
auto input = std::make_shared<core::Tensor>(
    core::Shape({8, 3, 224, 224}),  // batch=8
    core::DataType::FLOAT32
);

// Engine 会验证非 batch 维度是否匹配
outputs = engine.forward({{"input", input}});
```

### 形状验证

Engine 在 `forward()` 时会验证输入形状：

```cpp
// Engine::forward() 中的验证
for (const auto& input_name : graph_->inputs()) {
    const auto& expected_shape = ...;
    const auto& actual_shape = inputs[input_name]->shape();
    
    // 检查维度数量
    if (expected_shape.ndim() != actual_shape.ndim()) {
        return ERROR_INVALID_ARGUMENT;
    }
    
    // 检查非动态维度
    for (size_t i = 0; i < expected_shape.ndim(); ++i) {
        if (expected_shape[i] < 0 || i == 0) continue;  // Skip dynamic/batch dim
        
        if (expected_shape[i] != actual_shape[i]) {
            MI_LOG_ERROR("Shape mismatch: expected " + 
                        expected_shape.to_string() + 
                        ", got " + actual_shape.to_string());
            return ERROR_INVALID_ARGUMENT;
        }
    }
}
```

---

## 调试技巧

### 1. 启用详细日志

```cpp
config.enable_profiling = true;
```

输出每个节点的形状推断结果。

### 2. 检查特定节点的形状

```cpp
auto node = graph->get_node("conv1");
if (node && !node->output_tensors().empty()) {
    auto shape = node->output_tensors()[0]->shape();
    std::cout << "conv1 output shape: " << shape.to_string() << std::endl;
}
```

### 3. 形状推断失败的常见原因

#### 权重形状未设置

```cpp
// ❌ 错误：权重没有设置形状
auto weight = std::make_shared<core::Tensor>();
// 形状为空 []，导致推断失败

// ✓ 正确：设置权重形状
auto weight = std::make_shared<core::Tensor>(
    core::Shape({64, 3, 7, 7}),
    core::DataType::FLOAT32
);
```

#### 输入节点未设置形状

```cpp
// ❌ 错误：输入形状为空
auto input_node = graph->create_node("input");
// 没有设置 output_tensors

// ✓ 正确：设置输入形状
auto input_tensor = std::make_shared<core::Tensor>();
input_tensor->reshape(core::Shape({1, 3, 224, 224}));
input_node->set_output_tensors({input_tensor});
```

#### 图连接顺序错误

```cpp
// ❌ 错误：先创建依赖节点，再创建输入节点
auto conv = graph->create_node("conv");
auto input = graph->create_node("input");
graph->connect("input", "conv");  // conv 推断时 input 还没形状

// ✓ 正确：按拓扑序创建或让 Engine 排序
// Engine 会自动按拓扑序推断，不用担心顺序
```

---

## 示例程序

### 完整示例

参见 `examples/shape_inference_demo.cpp`：

```bash
# 编译
cmake --build build --target shape_inference_demo

# 运行
./build/Debug/bin/shape_inference_demo
```

**预期输出：**
```
========================================
Shape Inference Demo (TensorRT-style)
========================================

[Step 1] Building graph...
Graph built with 6 nodes

[Step 2] Before shape inference:
Node: input
  Outputs:
    [0] [1, 1, 28, 28]
Node: conv1
  Outputs:
    [0] []  ← 空形状

[Step 3] Building Engine (with shape inference)...
[Engine] Step 3: Inferring tensor shapes...
[Engine] Node conv1 output[0] shape: [1, 32, 28, 28]
[Engine] Node relu1 output[0] shape: [1, 32, 28, 28]
[Engine] Node pool1 output[0] shape: [1, 32, 14, 14]
[Engine] Node flatten output[0] shape: [1, 6272]
[Engine] Node fc output[0] shape: [1, 10]
[Engine] Shape inference completed: 5 tensor(s) inferred, 0 failed

[Step 4] After shape inference:
Node: conv1
  Outputs:
    [0] [1, 32, 28, 28]  ← 已推断
Node: fc
  Outputs:
    [0] [1, 10]

✓ All shapes inferred correctly!
```

### 单元测试

参见 `tests/test_shape_inference.cpp`：

```bash
# 运行测试
ctest -R test_shape_inference
```

测试覆盖：
- Conv2D 各种配置（padding, stride, dilation）
- Pooling（MaxPool, AvgPool）
- Linear（2D, 多维）
- ReLU（保持形状）
- Flatten（不同 axis）
- 错误处理（形状不匹配）

---

## 与内存规划的集成

形状推断是内存规划的前置步骤：

```cpp
Engine::build() {
    infer_shapes();       // 推断形状
    ↓
    plan_memory();        // 基于形状计算内存需求
    ↓
    allocate_tensors();   // 分配实际内存
}
```

**内存规划依赖准确的形状信息：**

```cpp
// memory_planner.cpp
for (const auto& node : nodes) {
    const auto& shape = node->output_tensors()[0]->shape();
    size_t bytes = shape.numel() * sizeof(float);  // 需要准确的 numel
    
    // 分析生命周期，复用内存
    analyze_lifetime(tensor_name, bytes);
}
```

**修复前后对比：**

| 阶段 | 修复前 | 修复后 |
|------|--------|--------|
| Shape 推断 | ❌ 未执行，形状为 `[]` | ✅ 自动推断 `[1,32,14,14]` |
| 内存计算 | ❌ `numel()=0`，使用兜底值 1024 bytes | ✅ 准确计算 `1*32*14*14*4 = 25KB` |
| 内存规划 | ⚠️ 统计不准 | ✅ 准确统计节省 50% |

---

## 最佳实践

### 1. 使用 ONNX 模型

✅ **推荐：**从 ONNX 导入，自动获取形状信息

```cpp
auto graph = parser.parse_from_file("model.onnx");
// 形状信息已包含在 ONNX 中
```

❌ **不推荐：**手动构建图，需要设置所有形状

### 2. 启用 Profiling 调试

```cpp
config.enable_profiling = true;  // 开发/调试时启用
```

### 3. 提前验证形状

```cpp
auto status = graph->validate();
if (status != core::Status::SUCCESS) {
    // 在 build 前检测问题
}
```

### 4. 处理动态维度

```cpp
// 定义时使用 -1
input_shape = core::Shape({-1, 3, 224, 224});

// 推理时提供实际值
input = Tensor(core::Shape({batch_size, 3, 224, 224}), ...);
```

---

## 动态 Shape 支持

Mini-Infer 支持基础的动态 Shape 推断：

✅ **已支持**：
- 动态 batch size（第 0 维可变）
- ONNX 动态维度自动识别（`dim_param`）
- Build 时使用默认值（batch=1）
- Forward 时允许不同 batch

⚠️ **限制**：
- 只支持动态 batch，其他维度必须固定
- 无运行时重推断
- 内存规划基于默认 batch

**详细说明**: [动态 Shape 支持文档](DYNAMIC_SHAPE_SUPPORT.md)

**示例**：
```cpp
// ONNX 模型定义: input = [-1, 3, 224, 224]
engine.build(graph);  // 使用 [1, 3, 224, 224]

// Forward 时可以使用不同 batch
auto input_b1 = make_shared<Tensor>(Shape({1, 3, 224, 224}));
engine.forward({{"input", input_b1}});  // ✅

auto input_b8 = make_shared<Tensor>(Shape({8, 3, 224, 224}));
engine.forward({{"input", input_b8}});  // ✅
```

---

## 相关文档

- **快速参考**: [QUICK_REFERENCE_SHAPE_INFERENCE.md](QUICK_REFERENCE_SHAPE_INFERENCE.md)
- **动态 Shape**: [DYNAMIC_SHAPE_SUPPORT.md](DYNAMIC_SHAPE_SUPPORT.md)
- **架构设计**: [SHAPE_INFERENCE_ARCHITECTURE.md](SHAPE_INFERENCE_ARCHITECTURE.md)
- **改进路线图**: [IMPROVEMENTS_ROADMAP.md](IMPROVEMENTS_ROADMAP.md)

---

## 总结

Mini-Infer 的 Shape 推断系统：

- ✅ **TensorRT 风格**：自动化、高效、易用
- ✅ **完整集成**：ONNX 导入 → Shape 推断 → 内存规划 → 分配
- ✅ **健壮**：详细的错误检测和日志
- ✅ **灵活**：支持动态 batch size

这是框架的核心功能之一，为后续的内存优化和推理执行提供了坚实基础！

