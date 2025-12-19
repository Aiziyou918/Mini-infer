# Mini-Infer API 文档 (API Reference)

## Core 模块

### Tensor

张量是 Mini-Infer 中的核心数据结构，采用了 View 机制，支持零拷贝操作。

#### 1. 创建张量
```cpp
// 方式1：创建新的 Tensor（分配内存）
core::Shape shape({1, 3, 224, 224});
auto tensor = core::Tensor::create(shape, core::DataType::FLOAT32);

// 方式2：绑定外部内存（Zero-Copy）
float* external_buffer = ...;
auto tensor_view = core::Tensor::create(shape, core::DataType::FLOAT32);
tensor_view->bind_external_data(external_buffer, size_in_bytes);
```

#### 2. 张量操作
```cpp
// 获取元数据
const core::Shape& shape = tensor->shape();
core::DataType dtype = tensor->dtype();
size_t bytes = tensor->size_in_bytes();
size_t offset = tensor->offset(); // 相对内存块的偏移量

// 获取数据指针
void* data = tensor->data();
float* float_data = tensor->data<float>();

// 形状操作 (Zero-Copy)
tensor->reshape({1, -1}); // 只修改 metadata，不移动数据

// 静态内存规划支持
tensor->bind_external_data_with_offset(storage, capacity, offset);
```

### Storage

`Storage` 类管理底层的物理内存块 (Raw Buffer)，通常由 Engine 管理，用户较少直接操作。

```cpp
auto storage = std::make_shared<core::Storage>(core::DeviceType::CPU);
storage->allocate(1024); // 分配 1KB
```

## Runtime 模块

Runtime 模块是推理的核心，采用了 "One Plan, Multiple Contexts" 架构。

### Engine

Engine 是也是一个门面类 (Facade)，负责构建 `InferencePlan`。

```cpp
// 1. 配置
runtime::EngineConfig config;
config.device_type = core::DeviceType::CPU;
config.enable_memory_planning = true; // 开启静态内存规划

// 2. 创建 Engine
runtime::Engine engine(config);

// 3. 构建 (Build Phase)
// 输入: 已经构建好的 Graph
core::Status status = engine.build(graph); 
```

### ExecutionContext

`ExecutionContext` 代表一次推理请求的状态，是线程不安全的，但可以从同一个 Engine 并发创建多个。

```cpp
// 1. 创建上下文
auto ctx = engine.create_context();

// 2. 准备数据
ctx->set_input("input_name", input_tensor);

// 3. 执行推理 (Run Phase)
engine.execute(ctx.get()); // 执行后结果存储在 ctx 中

// 4. 获取结果
auto output_tensor = ctx->get_output("output_name");
```

## Graph 模块

### Graph

Graph 管理计算图的拓扑结构，支持基于端口 (Port) 的连接。

```cpp
auto graph = std::make_shared<graph::Graph>();

// 1. 创建节点
auto node1 = graph->create_node("conv1");
auto node2 = graph->create_node("relu1");

// 2. 连接节点 (Port-based)
// connect(src_name, dst_name, src_port, dst_port)
graph->connect("conv1", "relu1", 0, 0); 

// 3. 设置图的输入输出
graph->set_inputs({"input"});
graph->set_outputs({"output"});
```

## Kernels & Registry 模块

这一层负责具体的算子计算分发。

```cpp
// 查找 Kernel
// Key: OpType + DeviceType + DataType
auto kernel = kernels::KernelRegistry::find(
    op_type, 
    core::DeviceType::CPU, 
    core::DataType::FLOAT32
);

// 执行 Kernel
if (kernel) {
    kernels::KernelContext ctx;
    ctx.inputs = ...;
    ctx.outputs = ...;
    kernel(ctx);
}
```

## Importers 模块

### OnnxParser

```cpp
importers::OnnxParser parser;

// 解析并返回 Graph
auto graph = parser.parse_from_file("model.onnx"); 
// 或者
auto graph = parser.parse_from_memory(buffer, size);

if (!graph) {
    std::cout << parser.get_error() << std::endl;
}
```

## 状态码 (Status)

所有主要 API 均返回 `core::Status`。

```cpp
if (status != core::Status::SUCCESS) {
    std::cerr << "Error: " << core::StatusString(status) << std::endl;
}
```
