# Shape 推断快速参考

## 一行代码启用

```cpp
Engine engine(config);
engine.build(graph);  // ← 自动完成 Shape 推断
```

就这么简单！Engine 在 build 时自动推断所有形状。

---

## 常见场景

### 场景 1: ONNX 模型

```cpp
// ✅ 最简单 - 形状信息已在 ONNX 中
auto graph = parser.parse_from_file("model.onnx");
engine.build(graph);  // 自动推断
```

### 场景 2: 手动构建图

```cpp
// 只需设置输入和权重的形状
auto input_tensor = std::make_shared<Tensor>();
input_tensor->reshape(Shape({1, 3, 224, 224}));  // ← 设置输入
input_node->set_output_tensors({input_tensor});

auto weight = std::make_shared<Tensor>(
    Shape({64, 3, 7, 7}),  // ← 设置权重
    DataType::FLOAT32
);
conv_node->set_input_tensors({nullptr, weight, bias});

// 中间层的形状会自动推断
engine.build(graph);
```

### 场景 3: 查看推断日志

```cpp
config.enable_profiling = true;  // ← 启用详细日志
engine.build(graph);

// 输出:
// [Engine] Node conv1 output[0] shape: [1, 64, 112, 112]
// [Engine] Node pool1 output[0] shape: [1, 64, 56, 56]
```

### 场景 4: 检查特定节点形状

```cpp
auto node = graph->get_node("conv1");
auto shape = node->output_tensors()[0]->shape();
std::cout << shape.to_string() << std::endl;  // [1, 64, 112, 112]
```

---

## 形状推断顺序

```
ONNX 解析
  ↓ (解析输入/权重形状)
Engine::build()
  ↓ (按拓扑序)
Node 1 → infer_shape() → [1, 64, 112, 112]
  ↓
Node 2 → infer_shape() → [1, 64, 56, 56]
  ↓
Node 3 → infer_shape() → [1, 1000]
```

---

## 常见错误

### ❌ 错误 1: 对空 Tensor 使用 reshape

```cpp
// ❌ 错误：reshape() 不能用于空 tensor
auto tensor = std::make_shared<Tensor>();
tensor->reshape(Shape({1, 3, 224, 224}));  // 失败！shape 仍为空

// ✅ 正确：创建时指定形状
auto tensor = std::make_shared<Tensor>(
    Shape({1, 3, 224, 224}),
    DataType::FLOAT32
);
```

**原因**: `reshape()` 要求新旧形状的元素数量相同，空 tensor 的 numel=0，所以无法 reshape。

### ❌ 错误 2: 权重没有形状

```cpp
// ❌ 错误
auto weight = std::make_shared<Tensor>();  // 形状为空

// ✅ 正确
auto weight = std::make_shared<Tensor>(
    Shape({64, 3, 7, 7}),
    DataType::FLOAT32
);
```

### ❌ 错误 3: 输入节点没有形状

```cpp
// ❌ 错误
auto input_node = graph->create_node("input");
// 没有设置 output_tensors

// ✅ 正确
auto input_tensor = std::make_shared<Tensor>(
    Shape({1, 3, 224, 224}),
    DataType::FLOAT32
);
input_node->set_output_tensors({input_tensor});
```

### ❌ 错误 4: 形状不匹配

```cpp
// ❌ 错误
// Conv weight: [64, 32, 3, 3]  需要 C_in=32
// 但输入是:  [1, 16, 224, 224]  实际 C_in=16

// 推断时会报错: ERROR_INVALID_ARGUMENT
```

---

## 算子形状规则

| 算子 | 输入形状 | 输出形状 | 公式 |
|-----|---------|---------|-----|
| **Conv2D** | [N,C_in,H,W] | [N,C_out,H',W'] | H' = (H+2P-K)/S+1 |
| **Pooling** | [N,C,H,W] | [N,C,H',W'] | H' = (H+2P-K)/S+1 |
| **Linear** | [...,in_f] | [...,out_f] | 保持前面维度 |
| **ReLU** | [任意] | [相同] | 不改变形状 |
| **Flatten** | [N,C,H,W] | [N,C*H*W] | 从 axis=1 展平 |
| **Reshape** | [...] | [...] | 总元素数相同 |

---

## 动态形状

### 当前支持情况

✅ **支持**：
- 动态 batch size（第 0 维）
- ONNX 动态维度自动识别
- Forward 时允许不同 batch

⚠️ **限制**：
- 只支持动态 batch，其他维度必须固定
- 内存规划基于默认 batch=1
- 无运行时重推断

详见：[动态 Shape 支持文档](DYNAMIC_SHAPE_SUPPORT.md)

### 使用方法

```cpp
// 1. ONNX 中定义动态维度
// input shape = [-1, 3, 224, 224]

// 2. Build 时使用默认 batch=1
engine.build(graph);  // 内部使用 [1, 3, 224, 224]

// 3. Forward 时可以使用不同 batch
auto input_batch1 = std::make_shared<Tensor>(
    Shape({1, 3, 224, 224}),  // ✅ batch=1
    DataType::FLOAT32
);
engine.forward({{"input", input_batch1}});

auto input_batch8 = std::make_shared<Tensor>(
    Shape({8, 3, 224, 224}),  // ✅ batch=8（允许）
    DataType::FLOAT32
);
engine.forward({{"input", input_batch8}});
```

### 注意事项

```cpp
// ✅ 允许：变化 batch size
Shape({1, 3, 224, 224});  // build
Shape({8, 3, 224, 224});  // forward - OK

// ❌ 不允许：变化其他维度
Shape({1, 3, 224, 224});  // build
Shape({1, 3, 256, 256});  // forward - ERROR!
```

---

## 调试技巧

### 1. 查看所有节点形状

```cpp
for (const auto& [name, node] : graph->nodes()) {
    if (!node->output_tensors().empty()) {
        auto shape = node->output_tensors()[0]->shape();
        std::cout << name << ": " << shape.to_string() << std::endl;
    }
}
```

### 2. 检查形状推断失败的节点

```cpp
config.enable_profiling = true;
engine.build(graph);

// 查看日志中的 WARNING:
// [Engine] Failed to infer shape for node: conv1
```

### 3. 验证输入形状

```cpp
// 在 forward 前检查
auto expected = graph->get_node("input")->output_tensors()[0]->shape();
auto actual = input_tensor->shape();
if (expected.to_string() != actual.to_string()) {
    // 形状不匹配
}
```

---

## 完整示例

```cpp
#include "mini_infer/importers/onnx_parser.h"
#include "mini_infer/runtime/engine.h"

int main() {
    // 1. 加载 ONNX 模型
    mini_infer::importers::OnnxParser parser;
    auto graph = parser.parse_from_file("model.onnx");
    
    // 2. 配置 Engine
    mini_infer::runtime::EngineConfig config;
    config.enable_graph_optimization = true;
    config.enable_memory_planning = true;
    config.enable_profiling = true;  // ← 查看形状推断过程
    
    // 3. 构建 Engine（自动推断形状）
    mini_infer::runtime::Engine engine(config);
    engine.build(graph);  // ← Shape 推断在这里发生
    
    // 4. 查看内存规划结果（基于准确的形状）
    const auto& plan = engine.get_memory_plan();
    std::cout << "Original memory: " 
              << plan.original_memory / 1024.0 << " KB\n";
    std::cout << "Optimized memory: " 
              << plan.total_memory / 1024.0 << " KB\n";
    std::cout << "Memory saving: " 
              << plan.memory_saving_ratio * 100.0f << "%\n";
    
    // 5. 运行推理
    auto input = std::make_shared<mini_infer::core::Tensor>(
        mini_infer::core::Shape({1, 3, 224, 224}),
        mini_infer::core::DataType::FLOAT32
    );
    
    auto outputs = engine.forward({{"input", input}});
    
    return 0;
}
```

---

## 更多信息

- **详细文档**: `docs/SHAPE_INFERENCE.md`
- **实现细节**: `docs/SHAPE_INFERENCE_IMPLEMENTATION.md`
- **示例程序**: `examples/shape_inference_demo.cpp`
- **单元测试**: `tests/test_shape_inference.cpp`

---

## 记住这些

✅ **自动完成** - Engine build 时自动推断所有形状  
✅ **ONNX 友好** - 自动解析 ONNX 模型的形状信息  
✅ **错误提前** - build 阶段检测形状错误，不是 forward 时  
✅ **详细日志** - enable_profiling 查看每个节点的形状  
✅ **内存准确** - 形状正确后，内存统计才准确  

就这么简单！🎉

