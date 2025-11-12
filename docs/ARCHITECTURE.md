# Mini-Infer 架构设计

## 总体架构

Mini-Infer 采用分层设计，从底层到上层依次为：

```
┌─────────────────────────────────────┐
│         Application Layer           │  应用层
├─────────────────────────────────────┤
│          Runtime Engine             │  运行时层
├─────────────────────────────────────┤
│      Graph & Optimization           │  图层
├─────────────────────────────────────┤
│          Operators                  │  算子层
├─────────────────────────────────────┤
│    Backend Abstraction Layer        │  后端抽象层
├──────────────┬──────────────────────┤
│  CPU Backend │    CUDA Backend      │  后端实现层
├──────────────┴──────────────────────┤
│         Core (Tensor, etc)          │  核心层
└─────────────────────────────────────┘
```

## 模块详解

### 1. Core 模块

**职责**: 提供框架的基础数据结构和类型定义

**主要组件**:
- `Tensor`: 多维数组，支持不同数据类型
- `Shape`: 形状表示
- `Allocator`: 内存分配器接口
- `DataType`: 数据类型枚举
- `Device`: 设备抽象

**设计原则**:
- 最小化依赖，不依赖其他模块
- 高性能的内存管理
- 类型安全

### 2. Backends 模块

**职责**: 提供硬件抽象层，支持多种计算后端

**架构**:
```
Backend (接口)
    ├── CPUBackend (CPU 实现)
    └── CUDABackend (GPU 实现, 待开发)
```

**关键接口**:
- 内存管理：allocate, deallocate
- 数据传输：memcpy, memset
- 同步控制：synchronize

**扩展性**:
- 通过继承 `Backend` 接口可以轻松添加新后端
- 后端工厂模式支持动态后端选择

### 3. Operators 模块

**职责**: 实现各种深度学习算子

**算子注册机制**:
```cpp
REGISTER_OPERATOR(Conv2D, Conv2DOperator);
```

**算子接口**:
- `forward()`: 前向计算
- `infer_shape()`: 形状推断

**已实现算子**:
- Conv2D (卷积)
- 更多算子开发中...

**待实现算子**:
- ReLU, Sigmoid, Tanh (激活函数)
- MaxPool, AvgPool (池化)
- BatchNorm, LayerNorm (归一化)
- Gemm, MatMul (矩阵运算)
- Concat, Split (张量操作)

### 4. Graph 模块

**职责**: 表示和管理计算图

**组件**:
- `Node`: 计算图节点，包含算子和张量
- `Graph`: 计算图，管理节点和边

**功能**:
- 图构建：创建节点、连接节点
- 拓扑排序：确定执行顺序
- 环检测：验证图的有效性
- 图优化（待完善）：算子融合、常量折叠等

**图表示**:
```
Input → Conv2D → ReLU → MaxPool → Output
```

### 5. Runtime 模块

**职责**: 执行推理，管理运行时状态

**核心组件**:
- `Engine`: 推理引擎
- `EngineConfig`: 引擎配置

**工作流程**:
1. 构建阶段：
   - 图验证
   - 图优化
   - 拓扑排序
   - 内存分配

2. 推理阶段：
   - 设置输入
   - 按拓扑顺序执行节点
   - 返回输出

**性能优化**（规划中）:
- 内存复用
- 算子融合
- 并行执行

### 6. Utils 模块

**职责**: 提供辅助工具

**组件**:
- `Logger`: 日志系统
- `Profiler`: 性能分析（待开发）
- `Timer`: 计时器（待开发）

## 数据流

### 推理数据流

```
User Input (Tensor)
    ↓
Engine::forward()
    ↓
Set Input Tensors
    ↓
Execute Nodes (Topological Order)
    ↓  Node1::forward()
    ↓  Node2::forward()
    ↓  ...
    ↓
Collect Output Tensors
    ↓
Return to User
```

### 内存管理流

```
Tensor Creation
    ↓
Allocator::allocate()
    ↓
Backend::allocate()
    ↓
System malloc/cudaMalloc
    ↓
Shared Pointer Management
    ↓
Automatic Deallocation
```

## 扩展点

### 1. 添加新算子

```cpp
// 1. 定义算子类
class MyOperator : public Operator {
public:
    Status forward(...) override;
    Status infer_shape(...) override;
};

// 2. 注册算子
REGISTER_OPERATOR(MyOp, MyOperator);
```

### 2. 添加新后端

```cpp
// 1. 实现 Backend 接口
class MyBackend : public Backend {
public:
    void* allocate(size_t size) override;
    // 实现其他接口...
};

// 2. 在 BackendFactory 中注册
```

### 3. 图优化

```cpp
// 在 Graph::optimize() 中添加优化 pass
Status Graph::optimize() {
    // Pass 1: 算子融合
    fuse_operators();
    
    // Pass 2: 常量折叠
    fold_constants();
    
    // Pass 3: 死代码消除
    eliminate_dead_code();
    
    return Status::SUCCESS;
}
```

## 性能考虑

### CPU 优化
- SIMD 指令（AVX, SSE）
- 多线程并行（OpenMP, TBB）
- 缓存友好的数据布局

### GPU 优化（未来）
- CUDA 内核优化
- 内存合并访问
- 共享内存使用
- Stream 并行

### 内存优化
- 内存池
- 原地操作
- 内存复用

## 依赖关系

```
Runtime → Graph → Operators → Backends → Core
         ↓                         ↓
       Utils                    Utils
```

## 线程安全

**当前状态**: 非线程安全

**未来计划**:
- 引擎级别的线程隔离
- 后端的线程池支持
- 算子的并行执行

## 错误处理

使用 `Status` 枚举进行错误处理：

```cpp
Status status = engine.build(graph);
if (status != Status::SUCCESS) {
    // 处理错误
    MI_LOG_ERROR("Build failed: " + status_to_string(status));
}
```

## 未来规划

### 短期目标
- [ ] 完善常用算子实现
- [ ] 实现基本的图优化
- [ ] 添加 ONNX 模型加载

### 中期目标
- [ ] CUDA 后端支持
- [ ] FP16/INT8 量化支持
- [ ] 动态 shape 支持

### 长期目标
- [ ] 自动调优
- [ ] 模型压缩
- [ ] 分布式推理

