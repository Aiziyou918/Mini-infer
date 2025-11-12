# Mini-Infer API 文档

## Core 模块

### Tensor

张量是 Mini-Infer 中的基础数据结构。

#### 创建张量

```cpp
// 方式1：使用静态工厂方法
core::Shape shape({1, 3, 224, 224});
auto tensor = core::Tensor::create(shape, core::DataType::FLOAT32);

// 方式2：直接构造
core::Tensor tensor(shape, core::DataType::FLOAT32);
```

#### 访问张量数据

```cpp
// 获取可变数据指针
void* data = tensor->data();
float* float_data = static_cast<float*>(data);

// 获取只读数据指针
const void* const_data = tensor->data();
```

#### 张量操作

```cpp
// 获取形状
const core::Shape& shape = tensor->shape();

// 获取数据类型
core::DataType dtype = tensor->dtype();

// 获取字节大小
size_t bytes = tensor->size_in_bytes();

// 重塑形状（元素数量必须相同）
core::Shape new_shape({1, 784});
tensor->reshape(new_shape);

// 检查是否为空
bool is_empty = tensor->empty();
```

### Shape

形状类用于表示张量的维度信息。

```cpp
// 创建形状
core::Shape shape({2, 3, 4, 5});

// 获取维度数
size_t ndim = shape.ndim();  // 4

// 获取总元素数
int64_t numel = shape.numel();  // 120

// 访问某个维度
int64_t dim0 = shape[0];  // 2

// 转换为字符串
std::string str = shape.to_string();  // "[2, 3, 4, 5]"
```

### DataType

支持的数据类型：

- `DataType::FLOAT32` - 32位浮点数
- `DataType::FLOAT16` - 16位浮点数
- `DataType::INT32` - 32位整数
- `DataType::INT8` - 8位整数
- `DataType::UINT8` - 8位无符号整数
- `DataType::BOOL` - 布尔型

## Backends 模块

### Backend

后端抽象接口，定义了不同硬件平台需要实现的方法。

```cpp
// 创建后端
auto backend = backends::BackendFactory::create_backend(core::DeviceType::CPU);

// 获取默认后端
auto backend = backends::BackendFactory::get_default_backend();

// 内存操作
void* ptr = backend->allocate(size);
backend->deallocate(ptr);
backend->memcpy(dst, src, size);
backend->memset(ptr, value, size);

// 同步
backend->synchronize();
```

### CPUBackend

CPU 后端实现。

```cpp
auto cpu_backend = backends::CPUBackend::get_instance();
```

## Operators 模块

### Operator

算子基类，定义了算子的通用接口。

```cpp
// 前向计算
core::Status forward(
    const std::vector<std::shared_ptr<core::Tensor>>& inputs,
    std::vector<std::shared_ptr<core::Tensor>>& outputs
);

// 推断输出形状
core::Status infer_shape(
    const std::vector<core::Shape>& input_shapes,
    std::vector<core::Shape>& output_shapes
);
```

### Conv2D

2D 卷积算子。

```cpp
// 创建 Conv2D 算子
auto conv = std::make_shared<operators::Conv2D>();

// 设置参数
auto param = std::make_shared<operators::Conv2DParam>();
param->kernel_h = 3;
param->kernel_w = 3;
param->stride_h = 1;
param->stride_w = 1;
param->padding_h = 1;
param->padding_w = 1;
conv->set_param(param);
```

## Graph 模块

### Graph

计算图类。

```cpp
// 创建图
graph::Graph graph;

// 创建节点
auto node1 = graph.create_node("node1");
auto node2 = graph.create_node("node2");

// 连接节点
graph.connect("node1", "node2");

// 设置输入输出
graph.set_inputs({"input"});
graph.set_outputs({"output"});

// 验证图
core::Status status = graph.validate();

// 拓扑排序
std::vector<std::shared_ptr<graph::Node>> sorted_nodes;
status = graph.topological_sort(sorted_nodes);

// 图优化
status = graph.optimize();
```

### Node

计算图节点。

```cpp
// 创建节点
graph::Node node("my_node");

// 设置算子
node.set_operator(op);

// 添加输入输出节点
node.add_input(input_node);
node.add_output(output_node);

// 设置张量
node.set_input_tensors(tensors);
node.set_output_tensors(tensors);
```

## Runtime 模块

### Engine

推理引擎。

```cpp
// 配置引擎
runtime::EngineConfig config;
config.device_type = core::DeviceType::CPU;
config.enable_profiling = true;
config.max_workspace_size = 1024 * 1024 * 1024;  // 1GB

// 创建引擎
runtime::Engine engine(config);

// 构建引擎
auto graph = std::make_shared<graph::Graph>();
// ... 构建图 ...
core::Status status = engine.build(graph);

// 执行推理
std::unordered_map<std::string, std::shared_ptr<core::Tensor>> inputs;
std::unordered_map<std::string, std::shared_ptr<core::Tensor>> outputs;
status = engine.forward(inputs, outputs);

// 获取输入输出信息
std::vector<std::string> input_names = engine.get_input_names();
std::vector<std::string> output_names = engine.get_output_names();

// 性能分析
engine.enable_profiling(true);
std::string profiling_info = engine.get_profiling_info();
```

## Utils 模块

### Logger

日志系统。

```cpp
// 获取日志实例
auto& logger = utils::Logger::get_instance();

// 设置日志级别
logger.set_level(utils::LogLevel::INFO);

// 使用日志宏
MI_LOG_DEBUG("Debug message");
MI_LOG_INFO("Info message");
MI_LOG_WARNING("Warning message");
MI_LOG_ERROR("Error message");
MI_LOG_FATAL("Fatal message");
```

## 状态码

```cpp
enum class Status {
    SUCCESS,                    // 成功
    ERROR_INVALID_ARGUMENT,     // 无效参数
    ERROR_OUT_OF_MEMORY,        // 内存不足
    ERROR_NOT_IMPLEMENTED,      // 未实现
    ERROR_RUNTIME,              // 运行时错误
    ERROR_BACKEND,              // 后端错误
    ERROR_UNKNOWN               // 未知错误
};

// 转换为字符串
const char* str = core::status_to_string(status);
```

