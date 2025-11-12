# Mini-Infer 快速入门

本文档将帮助你快速上手 Mini-Infer 推理框架。

## 第一步：构建项目

### Windows

```powershell
.\build.ps1 -Test
```

### Linux/macOS

```bash
chmod +x build.sh
./build.sh --test
```

## 第二步：运行第一个示例

### 创建张量

创建文件 `my_first_app.cpp`：

```cpp
#include "mini_infer/mini_infer.h"
#include <iostream>

using namespace mini_infer;

int main() {
    // 1. 创建一个张量
    core::Shape shape({1, 3, 224, 224});  // NCHW 格式
    auto tensor = core::Tensor::create(shape, core::DataType::FLOAT32);
    
    std::cout << "创建了形状为 " << tensor->shape().to_string() 
              << " 的张量" << std::endl;
    std::cout << "总元素数: " << tensor->shape().numel() << std::endl;
    std::cout << "内存大小: " << tensor->size_in_bytes() << " bytes" << std::endl;
    
    // 2. 访问和填充数据
    float* data = static_cast<float*>(tensor->data());
    for (int64_t i = 0; i < 10; ++i) {
        data[i] = static_cast<float>(i) * 0.1f;
        std::cout << "data[" << i << "] = " << data[i] << std::endl;
    }
    
    return 0;
}
```

编译运行：

```bash
# Linux/macOS
g++ my_first_app.cpp -o my_first_app \
    -Iinclude \
    -Lbuild/lib \
    -lmini_infer_core \
    -lmini_infer_utils \
    -lpthread \
    -std=c++17

./my_first_app

# Windows (使用 MSVC)
cl my_first_app.cpp /std:c++17 /Iinclude /link build\lib\Release\mini_infer_core.lib
```

## 第三步：使用后端

```cpp
#include "mini_infer/mini_infer.h"
#include <iostream>

using namespace mini_infer;

int main() {
    // 获取 CPU 后端
    auto backend = backends::BackendFactory::get_default_backend();
    
    std::cout << "使用后端: " << backend->name() << std::endl;
    
    // 分配内存
    size_t size = 1024 * sizeof(float);
    void* ptr = backend->allocate(size);
    
    // 初始化为 0
    backend->memset(ptr, 0, size);
    
    // 填充数据
    float* data = static_cast<float*>(ptr);
    for (int i = 0; i < 10; ++i) {
        data[i] = i * 0.5f;
    }
    
    // 释放内存
    backend->deallocate(ptr);
    
    std::cout << "内存操作完成" << std::endl;
    
    return 0;
}
```

## 第四步：构建计算图

```cpp
#include "mini_infer/mini_infer.h"
#include <iostream>

using namespace mini_infer;

int main() {
    // 创建计算图
    auto graph = std::make_shared<graph::Graph>();
    
    // 添加节点
    auto input = graph->create_node("input");
    auto conv1 = graph->create_node("conv1");
    auto relu1 = graph->create_node("relu1");
    auto pool1 = graph->create_node("pool1");
    auto output = graph->create_node("output");
    
    // 连接节点
    graph->connect("input", "conv1");
    graph->connect("conv1", "relu1");
    graph->connect("relu1", "pool1");
    graph->connect("pool1", "output");
    
    // 设置输入输出
    graph->set_inputs({"input"});
    graph->set_outputs({"output"});
    
    // 验证图
    auto status = graph->validate();
    if (status == core::Status::SUCCESS) {
        std::cout << "✓ 图验证通过" << std::endl;
    }
    
    // 拓扑排序
    std::vector<std::shared_ptr<graph::Node>> sorted_nodes;
    status = graph->topological_sort(sorted_nodes);
    
    if (status == core::Status::SUCCESS) {
        std::cout << "执行顺序: ";
        for (const auto& node : sorted_nodes) {
            std::cout << node->name() << " -> ";
        }
        std::cout << "完成" << std::endl;
    }
    
    return 0;
}
```

输出：
```
✓ 图验证通过
执行顺序: input -> conv1 -> relu1 -> pool1 -> output -> 完成
```

## 第五步：使用推理引擎

```cpp
#include "mini_infer/mini_infer.h"
#include <iostream>

using namespace mini_infer;

int main() {
    // 1. 构建计算图
    auto graph = std::make_shared<graph::Graph>();
    // ... 添加节点和连接 ...
    
    // 2. 配置引擎
    runtime::EngineConfig config;
    config.device_type = core::DeviceType::CPU;
    config.enable_profiling = true;
    
    // 3. 创建引擎
    runtime::Engine engine(config);
    
    // 4. 构建引擎（这会优化图并分配内存）
    auto status = engine.build(graph);
    if (status != core::Status::SUCCESS) {
        std::cerr << "引擎构建失败" << std::endl;
        return 1;
    }
    
    // 5. 准备输入数据
    core::Shape input_shape({1, 3, 224, 224});
    auto input_tensor = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    
    // 填充输入数据
    float* input_data = static_cast<float*>(input_tensor->data());
    for (int64_t i = 0; i < input_tensor->shape().numel(); ++i) {
        input_data[i] = 0.5f;  // 示例数据
    }
    
    // 6. 执行推理
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> inputs;
    inputs["input"] = input_tensor;
    
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> outputs;
    status = engine.forward(inputs, outputs);
    
    if (status == core::Status::SUCCESS) {
        std::cout << "✓ 推理成功" << std::endl;
        
        // 获取输出
        auto output_tensor = outputs["output"];
        std::cout << "输出形状: " << output_tensor->shape().to_string() << std::endl;
    }
    
    // 7. 查看性能信息（如果启用了 profiling）
    if (config.enable_profiling) {
        std::cout << engine.get_profiling_info() << std::endl;
    }
    
    return 0;
}
```

## 常用 API 速查

### 张量操作

```cpp
// 创建张量
auto tensor = core::Tensor::create({N, C, H, W}, core::DataType::FLOAT32);

// 访问数据
float* data = static_cast<float*>(tensor->data());

// 获取信息
tensor->shape()           // 形状
tensor->dtype()           // 数据类型
tensor->size_in_bytes()   // 字节大小
tensor->empty()           // 是否为空

// 重塑
tensor->reshape(new_shape);
```

### 图操作

```cpp
// 创建图和节点
auto graph = std::make_shared<graph::Graph>();
auto node = graph->create_node("node_name");

// 连接节点
graph->connect("src_node", "dst_node");

// 设置输入输出
graph->set_inputs({"input1", "input2"});
graph->set_outputs({"output"});

// 验证和排序
graph->validate();
graph->topological_sort(sorted_nodes);
```

### 引擎操作

```cpp
// 创建引擎
runtime::EngineConfig config;
config.device_type = core::DeviceType::CPU;
runtime::Engine engine(config);

// 构建和执行
engine.build(graph);
engine.forward(inputs, outputs);

// 获取信息
engine.get_input_names();
engine.get_output_names();
```

### 日志

```cpp
// 设置日志级别
utils::Logger::get_instance().set_level(utils::LogLevel::INFO);

// 使用日志
MI_LOG_DEBUG("调试信息");
MI_LOG_INFO("普通信息");
MI_LOG_WARNING("警告信息");
MI_LOG_ERROR("错误信息");
```

## 下一步

- 阅读 [API 文档](API.md) 了解详细接口
- 阅读 [架构文档](ARCHITECTURE.md) 了解设计原理
- 查看 `examples/` 目录下的更多示例
- 尝试实现自己的算子

## 常见问题

**Q: 如何在自己的 CMake 项目中使用 Mini-Infer？**

A: 在 CMakeLists.txt 中添加：

```cmake
add_subdirectory(path/to/Mini-Infer)
target_link_libraries(your_target PRIVATE mini_infer_runtime)
```

**Q: 支持哪些数据类型？**

A: 目前支持 FLOAT32, FLOAT16, INT32, INT8, UINT8, BOOL。

**Q: 如何启用 GPU 支持？**

A: GPU (CUDA) 支持正在开发中，敬请期待。

**Q: 性能如何优化？**

A: 
- 使用 Release 模式构建
- 启用编译器优化选项
- 使用合适的数据类型（如 FP16）
- 未来版本将支持图优化和算子融合

## 获取帮助

- 查看 [Issues](https://github.com/your-repo/Mini-Infer/issues)
- 阅读[贡献指南](../CONTRIBUTING.md)
- 查看代码注释和文档

祝你使用愉快！🚀

