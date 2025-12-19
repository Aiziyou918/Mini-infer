# Mini-Infer

一个轻量极大、高性能的深度学习推理框架，架构设计灵感源自 TensorRT 和 PyTorch。我们追求极致的 **Zero-Copy** 和 **Static Memory Planning**。

## 项目特性

- 🚄 **高性能 (High Performance)**:
    - **静态内存规划**: 采用 Linear Scan 算法，将所有中间张量压缩到一块连续内存中，极大降低碎片和分配开销。
    - **零拷贝**: Tensor View 设计，支持切片和 Reshape 而不产生数据拷贝。
    - **无锁 Allocator**: 针对高性能场景优化的内存分配器。
- 🧩 **模块化设计 (Modular Architecture)**:
    - **Core**: 基础数据结构 (Tensor/Storage)。
    - **Runtime**: 推理引擎 (InferencePlan/ExecutionContext)，支持并发推理。
    - **Backends**: 异构设备管理 (DeviceContext/Registry)，支持 CPU/CUDA 热插拔。
- 🔌 **ONNX 支持**:
    - 内置 ONNX 解析器，支持将 ONNX 模型直接导入为计算图。
    - **Port-Based Graph**: 支持多输入多输出 (MIMO) 的复杂拓扑结构。

## 项目结构

```
Mini-Infer/
├── include/mini_infer/
│   ├── core/          # 核心数据结构（Tensor, Storage, Allocator）
│   ├── backends/      # 执行环境（DeviceContext）
│   ├── kernels/       # 算子注册表与内核（Registry, Dispatcher）
│   ├── graph/         # 计DAG 图结构（Node, Edge, Port）
│   ├── runtime/       # 推理引擎（Plan, Context, MemoryPlanner）
│   └── importers/     # 模型导入（OnnxParser）
├── src/               # 源代码实现
└── examples/          # 示例代码
```

## 快速开始 (Quick Start)

### 1. 构建项目

推荐使用 **Conan** 自动管理依赖（如 Protobuf）：

```bash
# Windows (PowerShell) 一键构建
.\build.ps1

# Linux/macOS (Shell) 一键构建
./build.sh
```

或者手动使用 CMake：

```bash
mkdir build && cd build
# 启用 ONNX 与 Profiling
cmake .. -DMINI_INFER_ENABLE_ONNX=ON -DMINI_INFER_ENABLE_PROFILING=ON
cmake --build . --config Release
```

### 2. 运行示例 (Run Example)

我们提供了一个 LeNet-5 的完整示例：

```bash
# Windows
.\build\Release\bin\onnx_parser_example.exe .\models\python\lenet5\models\lenet5.onnx

# Linux
./build/bin/onnx_parser_example ./models/python/lenet5/models/lenet5.onnx
```

### 3. C++ API 示例

```cpp
#include "mini_infer/runtime/engine.h"
#include "mini_infer/importers/onnx_parser.h"

using namespace mini_infer;

int main() {
    // 1. 解析 ONNX
    importers::OnnxParser parser;
    auto graph = parser.parse_from_file("model.onnx");

    // 2. 配置引擎
    runtime::EngineConfig config;
    config.enable_memory_planning = true; // 开启静态内存规划
    runtime::Engine engine(config);

    // 3. 构建 Plan (Optimization + Memory Planning)
    engine.build(graph);

    // 4. 创建 Context 并执行
    auto ctx = engine.create_context();
    
    // 准备数据
    auto input_tensor = core::Tensor::create({1, 3, 224, 224});
    // ... fill data ...
    
    ctx->set_inputs({{"input", input_tensor}});
    engine.execute(ctx.get()); // 零拷贝执行

    // 获取结果
    auto output = ctx->get_output("output");
}
```

## 架构概览

查看 [ARCHITECTURE.md](docs/ARCHITECTURE.md) 获取详细的架构设计文档。

## 贡献

欢迎提交 Issue 和 Pull Request！我们正在积极寻找以下贡献：
- [ ] SIMD 优化 (AVX2/NEON) for CPU Kernels
- [ ] CUDA Kernels 实现
- [ ] 更多 ONNX 算子支持

## 许可证

MIT License
