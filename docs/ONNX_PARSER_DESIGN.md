# ONNX Parser Architecture Design

## 概述

Mini-Infer 的 ONNX 解析器采用模块化设计，参考 TensorRT 的架构理念，实现可扩展、易维护的模型导入系统。

## 核心组件

### 1. OnnxParser (主接口)
- **文件**: `onnx_parser.h/cpp`
- **职责**: 提供用户API，协调整个导入流程
- **接口**:
  ```cpp
  std::unique_ptr<graph::Graph> parse_from_file(const std::string& model_path);
  std::unique_ptr<graph::Graph> parse_from_buffer(const void* buffer, size_t size);
  ```

### 2. ModelImporter (模型导入器)
- **文件**: `model_importer.h/cpp`
- **职责**: 解析 ONNX ModelProto，协调图构建
- **功能**:
  - 解析模型元信息（IR版本、生产者信息）
  - 导入初始化器（权重）
  - 导入图节点（算子）
  - 构建图连接关系

### 3. OperatorImporter (算子导入器基类)
- **文件**: `operator_importer.h/cpp`
- **职责**: 定义算子导入接口
- **设计模式**: Strategy Pattern
- **接口**:
  ```cpp
  virtual core::Status import_operator(
      ImporterContext& ctx,
      const onnx::NodeProto& node
  ) = 0;
  ```

### 4. OperatorRegistry (算子注册表)
- **文件**: `operator_importer.h/cpp`
- **职责**: 管理算子导入器的注册和查找
- **设计模式**: Factory Pattern + Registry Pattern
- **功能**:
  - 动态注册算子导入器
  - 查找算子导入器
  - 获取支持的算子列表

### 5. ImporterContext (导入上下文)
- **文件**: `operator_importer.h/cpp`
- **职责**: 维护导入过程中的共享状态
- **包含**:
  - 正在构建的 Graph
  - Tensor 注册表（名称 -> Tensor 映射）
  - Weight 注册表（初始化器）
  - 错误管理
  - 日志系统

### 6. AttributeHelper (属性解析工具)
- **文件**: `attribute_helper.h/cpp`
- **职责**: 解析 ONNX 节点属性
- **功能**:
  - 类型安全的属性访问
  - 默认值支持
  - 数组属性处理

### 7. WeightImporter (权重导入器)
- **文件**: `weight_importer.h/cpp`
- **职责**: 将 ONNX TensorProto 转换为 Mini-Infer Tensor
- **功能**:
  - 数据类型转换
  - 原始数据导入
  - 类型化数据导入

### 8. Builtin Operators (内置算子)
- **文件**: `builtin_operators.h/cpp`
- **职责**: 实现常用 ONNX 算子的导入
- **已支持算子**:
  - Conv
  - Gemm / MatMul
  - Relu
  - MaxPool / AveragePool
  - BatchNormalization
  - Add / Mul
  - Reshape / Flatten
  - Concat / Softmax

## 工作流程

```
┌─────────────────┐
│  用户调用       │
│  OnnxParser     │
│  parse_from_*() │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ModelImporter  │
│  读取ONNX文件   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  解析模型信息   │
│  IR版本/生产者  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  导入初始化器   │
│  WeightImporter │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  逐节点导入     │
│  查找算子导入器 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OperatorImporter│
│  解析属性       │
│  创建算子       │
│  添加到图       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  返回构建好的   │
│  Graph          │
└─────────────────┘
```

## 算子导入流程

每个算子导入器遵循统一的流程：

1. **属性解析**: 使用 AttributeHelper 提取算子属性
2. **输入验证**: 检查输入数量和类型
3. **算子创建**: 创建对应的 Mini-Infer 算子
4. **节点添加**: 将算子添加到图中
5. **输出注册**: 注册输出 Tensor

示例（Conv 算子）:
```cpp
core::Status ConvImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    // 1. 解析属性
    AttributeHelper attrs(node);
    auto kernel_shape = attrs.get_ints("kernel_shape");
    auto strides = to_int_vector(attrs.get_ints("strides"));
    auto pads = to_int_vector(attrs.get_ints("pads"));
    
    // 2. 获取输入
    auto input = ctx.get_tensor(node.input(0));
    auto weight = ctx.get_weight(node.input(1));
    
    // 3. 创建算子
    auto conv_op = std::make_shared<operators::Conv2D>(...);
    
    // 4. 创建节点并添加到图
    auto graph_node = std::make_shared<graph::Node>(node.name(), conv_op);
    ctx.add_node(graph_node);
    
    // 5. 注册输出
    ctx.register_tensor(node.output(0), output_tensor);
    
    return core::Status::SUCCESS;
}
```

## 扩展性设计

### 添加新算子

1. 创建算子导入器类:
```cpp
class MyOpImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override {
        // 实现导入逻辑
    }
    const char* get_op_type() const override { return "MyOp"; }
};
```

2. 注册算子:
```cpp
void register_builtin_operators(OperatorRegistry& registry) {
    REGISTER_ONNX_OPERATOR("MyOp", MyOpImporter);
}
```

### 自定义算子

用户可以注册自定义算子:
```cpp
OnnxParser parser;
parser.get_registry().register_operator("CustomOp", []() {
    return std::make_unique<CustomOpImporter>();
});
```

## 错误处理

- **分层错误传播**: Status 返回码 + 错误消息
- **上下文错误记录**: ImporterContext 记录所有错误
- **详细日志**: 可选的详细日志输出
- **用户友好**: 清晰的错误信息和位置信息

## 与 TensorRT 的对比

| 特性 | Mini-Infer | TensorRT |
|-----|-----------|----------|
| **架构设计** | 模块化、清晰 | 模块化、复杂 |
| **算子数量** | 15+ (可扩展) | 300+ |
| **动态形状** | 计划中 | 完整支持 |
| **性能优化** | 基础 | 高度优化 |
| **易用性** | ✅ 简单易懂 | 学习曲线陡 |
| **可定制性** | ✅ 高度可定制 | 黑盒设计 |
| **适用场景** | 学习/轻量级 | 生产环境 |

## 下一步计划

### 短期 (P0)
- [x] 核心架构设计
- [ ] 实现核心算子（Conv, Gemm, ReLU等）
- [ ] 完整的 ModelImporter 实现
- [ ] 完整的 OnnxParser 实现
- [ ] CMake 集成

### 中期 (P1)
- [ ] 完善错误处理和日志
- [ ] 添加更多算子支持
- [ ] 图优化（常量折叠、层融合）
- [ ] 单元测试

### 长期 (P2)
- [ ] 动态形状支持
- [ ] 子图和控制流
- [ ] 量化算子
- [ ] 性能优化

## 参考资料

- [ONNX Specification](https://github.com/onnx/onnx/blob/main/docs/IR.md)
- [TensorRT ONNX Parser](https://github.com/NVIDIA/TensorRT/tree/main/parsers/onnx)
- [Mini-Infer Graph Architecture](../include/mini_infer/graph)
