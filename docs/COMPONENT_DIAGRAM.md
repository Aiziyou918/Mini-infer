# Mini-Infer 组件图 (Component Diagram)

本文档展示了 Mini-Infer 深度学习推理框架的组件架构。

## 1. 总体组件图

```mermaid
graph TB
    %% 用户层
    User[用户 User]
    
    %% Runtime 层
    Runtime[Runtime 运行时<br/>Engine, EngineConfig]
    
    %% 中间层组件
    Graph[Graph 计算图<br/>Node, Graph]
    Backends[Backends 后端<br/>Backend, CPUBackend]
    Operators[Operators 算子<br/>Operator, OpFactory]
    
    %% 核心层
    Core[Core 核心<br/>Tensor, Shape, DataType, Types, Allocator]
    
    %% 工具层
    Utils[Utils 工具<br/>Logger]
    
    %% 依赖关系
    User --> Runtime
    Runtime --> Graph
    Runtime --> Backends
    Runtime --> Operators
    Runtime --> Utils
    
    Graph --> Operators
    Graph --> Core
    
    Operators --> Backends
    Operators --> Core
    
    Backends --> Core
    
    %% 样式定义
    classDef userStyle fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef runtimeStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef componentStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef coreStyle fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef utilStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class User userStyle
    class Runtime runtimeStyle
    class Graph,Backends,Operators componentStyle
    class Core coreStyle
    class Utils utilStyle
```

## 2. 分层架构图

```mermaid
graph TD
    subgraph "应用层 Application Layer"
        A1[用户应用程序]
    end
    
    subgraph "运行时层 Runtime Layer"
        R1[Engine 推理引擎]
        R2[EngineConfig 引擎配置]
    end
    
    subgraph "图层 Graph Layer"
        G1[Graph 计算图]
        G2[Node 节点]
    end
    
    subgraph "算子层 Operators Layer"
        O1[Operator 算子基类]
        O2[OpFactory 算子工厂]
        O3[Conv2D 卷积]
        O4[其他算子...]
    end
    
    subgraph "后端层 Backends Layer"
        B1[Backend 后端接口]
        B2[CPUBackend CPU实现]
        B3[CUDABackend GPU实现]
    end
    
    subgraph "核心层 Core Layer"
        C1[Tensor 张量]
        C2[Shape 形状]
        C3[DataType 数据类型]
        C4[Allocator 内存分配器]
    end
    
    subgraph "工具层 Utils Layer"
        U1[Logger 日志系统]
    end
    
    A1 --> R1
    R1 --> G1
    R1 --> B1
    R1 --> U1
    G1 --> G2
    G1 --> O1
    O1 --> O2
    O1 --> O3
    O1 --> O4
    O1 --> B1
    O1 --> C1
    B1 --> B2
    B1 --> B3
    B1 --> C4
    C1 --> C2
    C1 --> C3
    C1 --> C4
    
    style A1 fill:#e1f5ff
    style R1,R2 fill:#fff3e0
    style G1,G2 fill:#f3e5f5
    style O1,O2,O3,O4 fill:#fce4ec
    style B1,B2,B3 fill:#e8eaf6
    style C1,C2,C3,C4 fill:#e8f5e9
    style U1 fill:#fff9c4
```

## 3. 模块依赖关系图

```mermaid
graph LR
    Runtime[Runtime<br/>运行时模块]
    Graph[Graph<br/>图模块]
    Operators[Operators<br/>算子模块]
    Backends[Backends<br/>后端模块]
    Core[Core<br/>核心模块]
    Utils[Utils<br/>工具模块]
    
    Runtime -->|依赖| Graph
    Runtime -->|依赖| Backends
    Runtime -->|依赖| Operators
    Runtime -->|依赖| Utils
    
    Graph -->|依赖| Operators
    Graph -->|依赖| Core
    
    Operators -->|依赖| Backends
    Operators -->|依赖| Core
    
    Backends -->|依赖| Core
    
    style Runtime fill:#ffccbc,stroke:#bf360c
    style Graph fill:#c5cae9,stroke:#283593
    style Operators fill:#b2dfdb,stroke:#004d40
    style Backends fill:#d1c4e9,stroke:#4527a0
    style Core fill:#a5d6a7,stroke:#1b5e20
    style Utils fill:#fff9c4,stroke:#f57f17
```

## 4. 数据流图

```mermaid
sequenceDiagram
    participant User as 用户
    participant Engine as Runtime::Engine
    participant Graph as Graph::Graph
    participant Node as Graph::Node
    participant Operator as Operators::Operator
    participant Backend as Backends::Backend
    participant Tensor as Core::Tensor
    
    User->>Engine: 1. 构建图 build(graph)
    Engine->>Graph: 2. 验证图
    Graph->>Graph: 3. 拓扑排序
    
    User->>Engine: 4. 推理 forward(inputs)
    Engine->>Engine: 5. 设置输入张量
    
    loop 遍历所有节点（拓扑顺序）
        Engine->>Node: 6. 执行节点
        Node->>Operator: 7. 调用算子 forward()
        Operator->>Backend: 8. 后端计算
        Backend->>Tensor: 9. 数据操作
        Tensor-->>Backend: 10. 返回结果
        Backend-->>Operator: 11. 返回
        Operator-->>Node: 12. 返回
        Node-->>Engine: 13. 返回
    end
    
    Engine->>Engine: 14. 收集输出
    Engine-->>User: 15. 返回结果
```

## 5. UML 组件图

```mermaid
C4Component
    title Mini-Infer 组件图

    Container_Boundary(runtime, "Runtime Layer 运行时层") {
        Component(engine, "Engine", "推理引擎", "管理推理流程")
        Component(config, "EngineConfig", "配置", "引擎配置参数")
    }

    Container_Boundary(graph_layer, "Graph Layer 图层") {
        Component(graph, "Graph", "计算图", "管理节点和边")
        Component(node, "Node", "节点", "图中的计算节点")
    }

    Container_Boundary(operators, "Operators Layer 算子层") {
        Component(operator, "Operator", "算子基类", "算子接口")
        Component(opfactory, "OpFactory", "算子工厂", "算子注册与创建")
        Component(conv2d, "Conv2D", "卷积算子", "2D卷积实现")
    }

    Container_Boundary(backends, "Backends Layer 后端层") {
        Component(backend, "Backend", "后端接口", "硬件抽象层")
        Component(cpu, "CPUBackend", "CPU后端", "CPU实现")
        Component(cuda, "CUDABackend", "CUDA后端", "GPU实现（待开发）")
    }

    Container_Boundary(core, "Core Layer 核心层") {
        Component(tensor, "Tensor", "张量", "多维数组")
        Component(shape, "Shape", "形状", "维度信息")
        Component(dtype, "DataType", "数据类型", "类型枚举")
        Component(allocator, "Allocator", "分配器", "内存管理")
    }

    Container_Boundary(utils, "Utils Layer 工具层") {
        Component(logger, "Logger", "日志", "日志系统")
    }

    Rel(engine, graph, "使用")
    Rel(engine, backend, "使用")
    Rel(engine, logger, "使用")
    Rel(graph, node, "包含")
    Rel(graph, operator, "使用")
    Rel(node, tensor, "使用")
    Rel(operator, backend, "使用")
    Rel(operator, tensor, "使用")
    Rel(backend, allocator, "使用")
    Rel(tensor, shape, "使用")
    Rel(tensor, dtype, "使用")
    Rel(tensor, allocator, "使用")
```

## 6. 组件接口说明

### Core 核心层
- **Tensor**: 多维数组，支持不同数据类型和设备
  - `create()`: 创建张量
  - `data()`: 获取数据指针
  - `shape()`: 获取形状
  - `dtype()`: 获取数据类型
- **Shape**: 张量形状表示
  - `dims()`: 获取维度数组
  - `size()`: 计算元素总数
- **DataType**: 数据类型枚举（FLOAT32, INT32等）
- **Allocator**: 内存分配器接口
  - `allocate()`: 分配内存
  - `deallocate()`: 释放内存

### Backends 后端层
- **Backend**: 硬件抽象接口
  - `allocate()`: 分配内存
  - `deallocate()`: 释放内存
  - `memcpy()`: 数据拷贝
  - `synchronize()`: 同步
- **CPUBackend**: CPU 后端实现
- **CUDABackend**: CUDA 后端实现（待开发）

### Operators 算子层
- **Operator**: 算子基类
  - `forward()`: 前向计算
  - `infer_shape()`: 形状推断
- **OpFactory**: 算子工厂，负责注册和创建算子
  - `create()`: 创建算子实例
  - `register_op()`: 注册算子
- **Conv2D**: 2D 卷积算子

### Graph 图层
- **Node**: 计算图节点
  - 包含算子
  - 管理输入输出张量
- **Graph**: 计算图
  - `create_node()`: 创建节点
  - `connect()`: 连接节点
  - `topological_sort()`: 拓扑排序
  - `validate()`: 验证图的有效性

### Runtime 运行时层
- **Engine**: 推理引擎
  - `build()`: 构建引擎
  - `forward()`: 执行推理
- **EngineConfig**: 引擎配置
  - 设备类型
  - 优化选项

### Utils 工具层
- **Logger**: 日志系统
  - 支持不同日志级别（DEBUG, INFO, WARNING, ERROR）
  - 格式化输出

## 7. 扩展点

```mermaid
graph TB
    subgraph "扩展机制"
        E1[添加新算子]
        E2[添加新后端]
        E3[添加图优化]
    end
    
    E1 --> |1. 继承 Operator| O[Operator 基类]
    E1 --> |2. 注册到| F[OpFactory]
    
    E2 --> |1. 继承 Backend| B[Backend 接口]
    E2 --> |2. 实现接口| I[allocate, memcpy, etc]
    
    E3 --> |1. 实现 Pass| P[Graph::optimize]
    E3 --> |2. 添加优化| T[算子融合、常量折叠等]
    
    style E1,E2,E3 fill:#ffecb3
    style O,F,B,I,P,T fill:#e1bee7
```

### 7.1 添加新算子示例

```cpp
// 1. 定义算子类
class ReLUOperator : public Operator {
public:
    Status forward(const std::vector<std::shared_ptr<Tensor>>& inputs,
                   std::vector<std::shared_ptr<Tensor>>& outputs,
                   Backend* backend) override {
        // 实现 ReLU 计算
        return Status::SUCCESS;
    }
    
    Status infer_shape(const std::vector<Shape>& input_shapes,
                       std::vector<Shape>& output_shapes) override {
        // 形状推断：ReLU 不改变形状
        output_shapes = input_shapes;
        return Status::SUCCESS;
    }
};

// 2. 注册算子
REGISTER_OPERATOR(ReLU, ReLUOperator);
```

### 7.2 添加新后端示例

```cpp
// 1. 实现 Backend 接口
class CUDABackend : public Backend {
public:
    void* allocate(size_t size) override {
        void* ptr = nullptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }
    
    void deallocate(void* ptr) override {
        cudaFree(ptr);
    }
    
    void memcpy(void* dst, const void* src, size_t size,
                MemcpyKind kind) override {
        // 实现 CUDA 内存拷贝
    }
    
    void synchronize() override {
        cudaDeviceSynchronize();
    }
};

// 2. 在 BackendFactory 中注册
```

## 8. 构建产物

```mermaid
graph LR
    subgraph "静态库"
        L1[libmini_infer_core.lib]
        L2[libmini_infer_backends.lib]
        L3[libmini_infer_operators.lib]
        L4[libmini_infer_graph.lib]
        L5[libmini_infer_runtime.lib]
        L6[libmini_infer_utils.lib]
    end
    
    subgraph "可执行文件"
        E1[test_tensor.exe]
        E2[test_backend.exe]
        E3[test_graph.exe]
        E4[simple_inference.exe]
        E5[build_graph.exe]
    end
    
    L1 --> L2
    L1 --> L3
    L2 --> L3
    L1 --> L4
    L3 --> L4
    L2 --> L5
    L3 --> L5
    L4 --> L5
    L6 --> L5
    
    L1 --> E1
    L2 --> E2
    L4 --> E3
    L5 --> E4
    L5 --> E5
    
    style L1,L2,L3,L4,L5,L6 fill:#c8e6c9
    style E1,E2,E3,E4,E5 fill:#bbdefb
```

## 9. 内存管理流程

```mermaid
graph TD
    A[创建 Tensor] --> B[Allocator::allocate]
    B --> C[Backend::allocate]
    C --> D{设备类型}
    D -->|CPU| E[std::malloc]
    D -->|CUDA| F[cudaMalloc]
    E --> G[返回指针]
    F --> G
    G --> H[shared_ptr 包装]
    H --> I[自动引用计数]
    I --> J{引用计数为0?}
    J -->|是| K[Allocator::deallocate]
    J -->|否| L[继续使用]
    K --> M[Backend::deallocate]
    M --> N[释放内存]
    
    style A fill:#e1f5ff
    style E,F fill:#ffccbc
    style H,I fill:#c8e6c9
    style N fill:#ffcdd2
```

## 10. 推理执行流程

```mermaid
flowchart TD
    Start([开始]) --> LoadGraph[加载计算图]
    LoadGraph --> BuildEngine[构建引擎 Engine::build]
    BuildEngine --> ValidateGraph{验证图}
    ValidateGraph -->|失败| Error1[返回错误]
    ValidateGraph -->|成功| TopoSort[拓扑排序]
    TopoSort --> AllocMem[分配内存]
    AllocMem --> SetInput[设置输入 Tensor]
    SetInput --> Forward[Engine::forward]
    
    Forward --> Loop{遍历节点}
    Loop -->|下一个节点| ExecNode[执行节点]
    ExecNode --> OpForward[Operator::forward]
    OpForward --> BackendComp[Backend 计算]
    BackendComp --> Loop
    Loop -->|完成| CollectOutput[收集输出]
    CollectOutput --> Return([返回结果])
    
    Error1 --> End([结束])
    Return --> End
    
    style Start fill:#e8f5e9
    style End fill:#e8f5e9
    style Error1 fill:#ffcdd2
    style BackendComp fill:#bbdefb
    style Return fill:#c8e6c9
```

---

## 说明

- **实线箭头**: 表示依赖关系（uses/depends on）
- **虚线箭头**: 表示创建关系（creates）
- **包含关系**: 表示模块包含子组件

## 如何查看

这些图使用 Mermaid 语法编写，可以在以下环境中渲染：

1. **GitHub**: 直接在 GitHub 上查看此 Markdown 文件
2. **VS Code**: 安装 Mermaid 预览插件
   - 推荐插件: "Markdown Preview Mermaid Support"
3. **在线工具**: 使用 [Mermaid Live Editor](https://mermaid.live/)
4. **文档生成**: MkDocs, GitBook 等支持 Mermaid 的文档工具

## 相关文档

- [架构设计文档](ARCHITECTURE.md) - 详细的架构说明
- [API 文档](API.md) - API 参考
- [构建指南](BUILD.md) - 如何构建项目
- [快速入门](GETTING_STARTED.md) - 入门教程

---

最后更新: 2025-11-16
