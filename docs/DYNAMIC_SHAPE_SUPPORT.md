# 动态 Shape 支持说明

## 📊 当前支持情况

Mini-Infer **已经支持**基础的动态 Shape 推断，但有一定限制。

---

## ✅ 已实现的功能

### 1. 动态维度识别

```cpp
// Shape 类支持动态维度（-1）
bool Shape::is_dynamic() const {
    for (int64_t dim : dims_) {
        if (dim < 0) {  // -1 表示动态维度
            return true;
        }
    }
    return false;
}
```

**示例**：
```cpp
Shape static_shape({1, 3, 224, 224});    // 静态形状
Shape dynamic_shape({-1, 3, 224, 224});  // 动态 batch size

assert(!static_shape.is_dynamic());  // false
assert(dynamic_shape.is_dynamic());  // true
```

### 2. ONNX 动态维度导入

从 ONNX 模型导入时自动识别动态维度：

```cpp
// src/importers/model_importer.cpp
if (dim.has_dim_param()) {
    // Dynamic dimension (e.g., batch size)
    dims.push_back(-1);
    ctx.log_info("  Dynamic dimension: " + dim.dim_param());
}
```

**ONNX 模型示例**：
```protobuf
input {
  name: "input"
  type {
    tensor_type {
      shape {
        dim { dim_param: "batch" }  # 动态维度
        dim { dim_value: 3 }        # 固定维度
        dim { dim_value: 224 }
        dim { dim_value: 224 }
      }
    }
  }
}
```

**导入结果**：
```cpp
// 解析为 Shape([-1, 3, 224, 224])
```

### 3. Build 时使用默认值

在 `Engine::build()` 阶段，动态维度使用默认值（batch=1）：

```cpp
// src/importers/model_importer.cpp
for (size_t j = 0; j < input_shape.ndim(); ++j) {
    int64_t dim = input_shape[j];
    if (dim < 0) {
        // Use batch size 1 as default for dynamic dimensions
        concrete_dims.push_back(1);
    } else {
        concrete_dims.push_back(dim);
    }
}
```

**原因**：
- Shape 推断需要具体的维度值
- 内存规划需要计算准确的 tensor size
- 使用 batch=1 作为合理的默认值

### 4. Forward 时支持不同 Batch Size

在 `Engine::forward()` 时允许不同的 batch size：

```cpp
// src/runtime/engine.cpp - validate_input_shapes()
for (size_t i = 0; i < expected_shape.ndim(); ++i) {
    // Skip dynamic dimensions (-1) or batch dimension (index 0)
    if (expected_shape[i] < 0 || i == 0) continue;  // ← 跳过 batch 维度
    
    if (expected_shape[i] != actual_shape[i]) {
        // 报错：其他维度必须匹配
    }
}
```

**示例**：
```cpp
// Build 时使用默认 batch=1
engine.build(graph);  // 内部使用 [1, 3, 224, 224]

// Forward 时可以使用不同 batch
auto input_batch1 = std::make_shared<Tensor>(
    Shape({1, 3, 224, 224}),  // ✅ batch=1
    DataType::FLOAT32
);
engine.forward({{"input", input_batch1}});

auto input_batch8 = std::make_shared<Tensor>(
    Shape({8, 3, 224, 224}),  // ✅ batch=8 (允许)
    DataType::FLOAT32
);
engine.forward({{"input", input_batch8}});
```

---

## ⚠️ 当前限制

### 1. 只支持 Batch 维度动态

**支持**：
```cpp
Shape({-1, 3, 224, 224});  // ✅ 动态 batch size
```

**不支持**：
```cpp
Shape({1, 3, -1, -1});     // ❌ 动态 H/W（未完全测试）
Shape({1, -1, 224, 224});  // ❌ 动态 channel（未完全测试）
```

**原因**：
- 当前只在第 0 维（batch）做了特殊处理
- 其他动态维度需要更复杂的形状推断逻辑

### 2. 内存规划基于默认 Batch

内存规划在 build 阶段完成，使用 batch=1：

```cpp
// Build 时
engine.build(graph);  // 使用 [1, 3, 224, 224] 计算内存

// 内存规划结果
Memory plan: 150KB (based on batch=1)
```

**影响**：
- Forward 时使用 batch=8，实际需要内存 = 150KB * 8 = 1200KB
- 内存池大小是按 batch=1 分配的（可能需要重新分配）

### 3. 没有运行时重新推断

Forward 时不会重新执行 shape 推断：

```cpp
// Forward 时
engine.forward({{"input", input_batch8}});  
// ❌ 不会重新推断每个节点的输出形状
// ✅ 直接使用 build 时推断的形状（可能导致尺寸不匹配）
```

---

## 📋 支持级别总结

| 功能 | 状态 | 说明 |
|-----|------|------|
| **识别动态维度** | ✅ 完全支持 | `Shape::is_dynamic()` |
| **ONNX 动态维度导入** | ✅ 完全支持 | 自动解析 `dim_param` |
| **Build 时默认值** | ✅ 完全支持 | 动态维度使用 1 |
| **Forward 时验证** | ✅ 部分支持 | 跳过 batch 维度检查 |
| **动态 Batch** | ✅ 基础支持 | 允许不同 batch size |
| **动态 H/W/C** | ⚠️ 未测试 | 理论可行，未验证 |
| **运行时重推断** | ❌ 不支持 | Forward 不重新推断形状 |
| **动态内存分配** | ⚠️ 部分支持 | 可能需要重新分配 |

---

## 🎯 使用建议

### 场景 1: 固定输入尺寸（推荐）

```cpp
// ONNX 模型有动态维度
// input: [-1, 3, 224, 224]

// Build 时使用默认 batch=1
engine.build(graph);

// Forward 时使用相同尺寸
auto input = std::make_shared<Tensor>(
    Shape({1, 3, 224, 224}),  // 与 build 时一致
    DataType::FLOAT32
);
engine.forward({{"input", input}});
```

**优点**：
- ✅ 性能最优
- ✅ 内存规划准确
- ✅ 无额外开销

### 场景 2: 变化的 Batch Size

```cpp
// Build 时使用默认 batch=1
engine.build(graph);

// Forward 时使用不同 batch
for (int batch : {1, 2, 4, 8}) {
    auto input = std::make_shared<Tensor>(
        Shape({batch, 3, 224, 224}),
        DataType::FLOAT32
    );
    engine.forward({{"input", input}});
}
```

**优点**：
- ✅ 灵活性高
- ✅ 支持动态 batch

**缺点**：
- ⚠️ 内存可能需要重新分配
- ⚠️ 可能有性能损失

### 场景 3: 完全动态形状（不推荐）

```cpp
// 任意尺寸输入
auto input1 = std::make_shared<Tensor>(Shape({1, 3, 224, 224}));
auto input2 = std::make_shared<Tensor>(Shape({4, 3, 512, 512}));
```

**现状**：
- ❌ 不支持运行时重推断
- ❌ 可能导致错误或崩溃

---

## 🚀 TensorRT 对比

### TensorRT 的动态 Shape 支持

```cpp
// TensorRT API
builder->setMaxBatchSize(32);
profile->setDimensions("input", 
    OptProfileSelector::kMIN, Dims4{1, 3, 224, 224});
profile->setDimensions("input", 
    OptProfileSelector::kMAX, Dims4{32, 3, 224, 224});
profile->setDimensions("input", 
    OptProfileSelector::kOPT, Dims4{8, 3, 224, 224});
```

**TensorRT 特性**：
1. ✅ 定义多个 Optimization Profile
2. ✅ 指定 Min/Max/Opt 范围
3. ✅ 运行时在范围内动态分配
4. ✅ 完整的动态维度支持（任意维度）

### Mini-Infer vs TensorRT

| 特性 | Mini-Infer | TensorRT |
|-----|-----------|----------|
| 动态 batch | ✅ 基础支持 | ✅ 完全支持 |
| 动态 H/W | ⚠️ 未测试 | ✅ 完全支持 |
| Optimization Profile | ❌ 不支持 | ✅ 支持 |
| 运行时重推断 | ❌ 不支持 | ✅ 支持 |
| 内存池动态调整 | ⚠️ 有限 | ✅ 完全支持 |

---

## 📝 未来改进方向

### 优先级 1: 运行时 Shape 重推断

```cpp
class Engine {
public:
    // 新增：运行时形状推断
    Status infer_shapes_runtime(
        const std::map<std::string, std::shared_ptr<core::Tensor>>& inputs
    ) {
        // 基于实际输入重新推断所有节点的输出形状
        // 更新内存分配
    }
    
    TensorMap forward(const TensorMap& inputs) override {
        // 1. 检查输入形状是否变化
        if (input_shape_changed(inputs)) {
            // 2. 重新推断形状
            infer_shapes_runtime(inputs);
            // 3. 重新分配内存（如需要）
            reallocate_if_needed();
        }
        // 4. 执行推理
        return execute(inputs);
    }
};
```

### 优先级 2: Optimization Profile

```cpp
struct OptimizationProfile {
    std::string input_name;
    Shape min_shape;
    Shape max_shape;
    Shape opt_shape;
};

class Engine {
public:
    void add_optimization_profile(const OptimizationProfile& profile);
    void set_active_profile(int index);
};
```

### 优先级 3: 动态内存池

```cpp
class DynamicMemoryPool {
public:
    // 根据实际形状动态调整内存池大小
    void resize(size_t new_size);
    
    // 记录最大使用量
    size_t peak_usage() const;
};
```

---

## 🧪 测试示例

### 测试 1: 动态 Batch 推理

```cpp
#include "mini_infer/importers/onnx_parser.h"
#include "mini_infer/runtime/engine.h"

int main() {
    // 1. 加载模型（假设有动态 batch）
    OnnxParser parser;
    auto graph = parser.parse_from_file("model.onnx");
    // input shape: [-1, 3, 224, 224]
    
    // 2. Build engine（使用默认 batch=1）
    EngineConfig config;
    Engine engine(config);
    engine.build(graph);
    
    // 3. 测试不同 batch size
    for (int batch : {1, 2, 4, 8}) {
        auto input = std::make_shared<Tensor>(
            Shape({batch, 3, 224, 224}),
            DataType::FLOAT32
        );
        
        std::cout << "Testing batch=" << batch << std::endl;
        auto outputs = engine.forward({{"input", input}});
        
        // 验证输出形状
        for (const auto& [name, tensor] : outputs) {
            std::cout << "  " << name << ": " 
                     << tensor->shape().to_string() << std::endl;
        }
    }
    
    return 0;
}
```

### 测试 2: 形状验证

```cpp
TEST(DynamicShapeTest, BatchDimensionAllowed) {
    // Build 时: [1, 3, 224, 224]
    auto graph = create_test_graph();
    Engine engine(config);
    engine.build(graph);
    
    // Forward 时: [8, 3, 224, 224]
    auto input = std::make_shared<Tensor>(
        Shape({8, 3, 224, 224}),
        DataType::FLOAT32
    );
    
    auto outputs = engine.forward({{"input", input}});
    EXPECT_TRUE(outputs.size() > 0);  // ✅ 应该成功
}

TEST(DynamicShapeTest, OtherDimensionsMustMatch) {
    // Build 时: [1, 3, 224, 224]
    auto graph = create_test_graph();
    Engine engine(config);
    engine.build(graph);
    
    // Forward 时: [1, 3, 256, 256] (H/W 不匹配)
    auto input = std::make_shared<Tensor>(
        Shape({1, 3, 256, 256}),
        DataType::FLOAT32
    );
    
    auto outputs = engine.forward({{"input", input}});
    EXPECT_TRUE(outputs.empty());  // ❌ 应该失败
}
```

---

## ✅ 总结

### 当前状态

**Mini-Infer 已经支持基础的动态 Shape 推断**：

✅ **支持**：
- 动态维度识别（-1）
- ONNX 动态维度导入
- 动态 batch size（第 0 维）
- Build 时使用默认值
- Forward 时 batch 维度灵活

⚠️ **限制**：
- 只充分测试了动态 batch
- 没有运行时重推断
- 内存规划基于默认值
- 不支持 Optimization Profile

### 推荐用法

```cpp
// ✅ 推荐：固定尺寸或只变化 batch
engine.build(graph);  // batch=1
engine.forward({{"input", input_batch1}});  // OK
engine.forward({{"input", input_batch8}});  // OK

// ⚠️ 谨慎：变化其他维度
engine.forward({{"input", input_different_hw}});  // 可能失败

// ❌ 不推荐：完全动态
// 需要实现运行时重推断
```

### 与 TensorRT 差距

Mini-Infer 提供了**基础的动态 Shape 支持**，适合：
- 固定输入尺寸的场景
- 只需要动态 batch 的场景

如需完整的动态 Shape 支持（如 TensorRT），需要实现：
1. 运行时 shape 重推断
2. Optimization Profile 机制
3. 动态内存池管理

这些功能可以作为未来的改进方向！🚀


