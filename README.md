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

### 1. 安装依赖

```bash
# 安装 Conan 包管理器
pip install conan

# 检测默认配置
conan profile detect --force
```

### 2. 构建项目

使用 **Conan** 自动管理依赖并构建：

```bash
# Windows (PowerShell)
# 步骤 1: 安装依赖并生成 CMake 预设
conan install . --output-folder=build --build=missing -s build_type=Release

# 步骤 2: 配置 CMake
cmake --preset conan-release

# 步骤 3: 编译
cmake --build --preset conan-release

# Linux/macOS (Bash)
# 步骤 1: 安装依赖并生成 CMake 预设
conan install . --output-folder=build --build=missing -s build_type=Release

# 步骤 2: 配置 CMake
cmake --preset conan-release

# 步骤 3: 编译
cmake --build --preset conan-release
```

### 3. 运行示例 (Run Example)

我们提供了一个 LeNet-5 的完整示例：

```bash
# Windows
.\build\Release\bin\onnx_parser_example.exe .\models\python\lenet5\models\lenet5.onnx

# Linux/macOS
./build/Release/bin/onnx_parser_example ./models/python/lenet5/models/lenet5.onnx
```

### 4. C++ API 示例

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
    
    ctx->set_input("input", input_tensor);
    engine.execute(ctx.get()); // 零拷贝执行

    // 获取结果
    auto output = ctx->get_output("output");
}
```

### 5. 构建选项

Conan 提供了灵活的构建选项：

```bash
# 启用 CUDA 支持
conan install . --output-folder=build --build=missing \
  -o enable_cuda=True \
  -o cuda_toolkit_root="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3"

# 禁用 ONNX 支持（减小二进制大小）
conan install . --output-folder=build --build=missing \
  -o enable_onnx=False

# 禁用日志（生产环境优化）
conan install . --output-folder=build --build=missing \
  -o enable_logging=False
```

详细的构建选项请参考 [Conan 构建指南](docs/CONAN_BUILD_GUIDE.md)。

## 📚 文档

- **[快速开始](QUICK_START.md)** - 快速上手指南
- **[Conan 构建指南](docs/CONAN_BUILD_GUIDE.md)** - 详细的 Conan 使用说明
- **[CUDA 配置指南](docs/CUDA_CONAN_SETUP.md)** - CUDA 后端配置
- **[架构设计](docs/ARCHITECTURE.md)** - 详细的架构设计文档
- **[API 文档](docs/API.md)** - API 参考手册
- **[入门教程](docs/GETTING_STARTED.md)** - 完整的入门教程

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！我们正在积极寻找以下贡献：
- [ ] SIMD 优化 (AVX2/NEON) for CPU Kernels
- [ ] CUDA Kernels 实现
- [ ] 更多 ONNX 算子支持

## 📄 许可证

MIT License
