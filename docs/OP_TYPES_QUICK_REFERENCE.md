# 算子类型管理 - 快速参考

## 🎯 核心决策

**使用String + 常量集中管理，不使用Enum**

## 📖 使用指南

### 1. 包含头文件

```cpp
#include "mini_infer/core/op_types.h"

using namespace mini_infer::op_types;
```

### 2. 使用常量替代魔法字符串

#### ❌ 不推荐（魔法字符串）

```cpp
if (node->get_operator()->name() == "Conv") {
    // ...
}

REGISTER_OPERATOR("ReLU", ReLU);
```

#### ✅ 推荐（使用常量）

```cpp
if (node->get_operator()->name() == kConv2D) {
    // ...
}

REGISTER_OPERATOR(kReLU, ReLU);
```

### 3. 使用辅助函数

```cpp
// 检查是否是激活函数
if (is_activation(op_name)) {
    // ...
}

// 检查是否是卷积
if (is_convolution(op_name)) {
    // ...
}

// 检查是否是池化
if (is_pooling(op_name)) {
    // ...
}
```

## 📋 常用常量

### 卷积
- `kConv2D` - "Conv"
- `kConvTranspose` - "ConvTranspose"

### 激活函数
- `kReLU` - "Relu"
- `kSigmoid` - "Sigmoid"
- `kTanh` - "Tanh"
- `kLeakyReLU` - "LeakyRelu"

### 池化
- `kMaxPool` - "MaxPool"
- `kAveragePool` - "AveragePool"
- `kGlobalAveragePool` - "GlobalAveragePool"

### 归一化
- `kBatchNorm` - "BatchNormalization"
- `kInstanceNorm` - "InstanceNormalization"
- `kLayerNorm` - "LayerNormalization"

### 线性运算
- `kGemm` - "Gemm"
- `kMatMul` - "MatMul"
- `kLinear` - "Linear"

### 形状操作
- `kReshape` - "Reshape"
- `kFlatten` - "Flatten"
- `kTranspose` - "Transpose"
- `kConcat` - "Concat"
- `kSplit` - "Split"

### 元素运算
- `kAdd` - "Add"
- `kSub` - "Sub"
- `kMul` - "Mul"
- `kDiv` - "Div"

## 🔧 添加新算子

### 1. 在op_types.h中添加常量

```cpp
// include/mini_infer/core/op_types.h
namespace mini_infer {
namespace op_types {

/// @brief My New Operator (ONNX: "MyNewOp")
constexpr const char* kMyNewOp = "MyNewOp";

} // namespace op_types
} // namespace mini_infer
```

### 2. 实现算子

```cpp
// src/operators/my_new_op.cpp
#include "mini_infer/core/op_types.h"

class MyNewOp : public Operator {
public:
    const char* name() const override { 
        return op_types::kMyNewOp; 
    }
    // ...
};

REGISTER_OPERATOR(op_types::kMyNewOp, MyNewOp);
```

### 3. 在FusionPass中使用

```cpp
// src/graph/fusion_pass.cpp
#include "mini_infer/core/op_types.h"

if (node->get_operator()->name() == op_types::kMyNewOp) {
    // ...
}
```

## ❓ FAQ

### Q: 为什么不用Enum？

**A**: 
- Enum会破坏可扩展性（用户无法自定义算子）
- 需要维护String→Enum映射表
- 字符串比较的性能损耗可忽略（仅在构建期）

### Q: 性能会受影响吗？

**A**: 
- 字符串比较只发生在**构建期**（解析、优化）
- **运行时**使用虚函数调用，不涉及字符串
- 实测开销<0.1%，完全可忽略

### Q: 如何防止拼写错误？

**A**: 
- 使用`op_types::kConv2D`而非`"Conv"`
- IDE会自动补全
- 拼写错误会导致编译失败

### Q: 自定义算子怎么办？

**A**: 
```cpp
// 用户代码（不需要修改框架）
class MyCustomOp : public Operator {
    const char* name() const override { 
        return "MyCustomOp";  // 直接使用字符串
    }
};

REGISTER_OPERATOR("MyCustomOp", MyCustomOp);
```

## 📚 相关文档

- 详细架构分析: `docs/OP_TYPES_ARCHITECTURE.md`
- API参考: `include/mini_infer/core/op_types.h`

---

**记住**: 使用常量，避免魔法字符串！✨
