# Mini-Infer 项目结构说明

本文档详细说明了 Mini-Infer 项目的目录结构和文件组织。

## 顶层目录结构

```
Mini-Infer/
├── CMakeLists.txt          # 顶层 CMake 配置文件
├── CMakePresets.json       # CMake 预设配置
├── README.md               # 项目说明文档
├── LICENSE                 # MIT 许可证
├── CONTRIBUTING.md         # 贡献指南
├── PROJECT_STRUCTURE.md    # 本文件
├── .gitignore             # Git 忽略文件配置
├── .clang-format          # 代码格式化配置
├── build.ps1              # Windows PowerShell 构建脚本
├── build.sh               # Linux/macOS Shell 构建脚本
├── include/               # 公共头文件目录
├── src/                   # 源代码实现目录
├── tests/                 # 测试代码目录
├── examples/              # 示例代码目录
└── docs/                  # 文档目录
```

## include/ - 头文件目录

公共 API 头文件，按模块组织：

```
include/mini_infer/
├── mini_infer.h           # 主头文件，包含所有公共 API
├── core/                  # 核心模块头文件
│   ├── tensor.h          # 张量类
│   ├── allocator.h       # 内存分配器
│   └── types.h           # 基础类型定义
├── backends/              # 后端模块头文件
│   ├── backend.h         # 后端抽象接口
│   └── cpu_backend.h     # CPU 后端实现
├── operators/             # 算子模块头文件
│   ├── operator.h        # 算子基类
│   └── conv2d.h          # Conv2D 算子
├── graph/                 # 计算图模块头文件
│   ├── node.h            # 图节点
│   └── graph.h           # 计算图
├── runtime/               # 运行时模块头文件
│   └── engine.h          # 推理引擎
└── utils/                 # 工具模块头文件
    └── logger.h          # 日志系统
```

## src/ - 源代码目录

实现文件，与头文件结构对应：

```
src/
├── core/                  # 核心模块实现
│   ├── CMakeLists.txt    # 模块构建配置
│   ├── tensor.cpp        # 张量实现
│   ├── allocator.cpp     # 内存分配器实现
│   └── types.cpp         # 类型定义实现
├── backends/              # 后端模块实现
│   ├── CMakeLists.txt
│   ├── backend.cpp       # 后端工厂实现
│   └── cpu_backend.cpp   # CPU 后端实现
├── operators/             # 算子模块实现
│   ├── CMakeLists.txt
│   ├── operator.cpp      # 算子注册机制
│   └── conv2d.cpp        # Conv2D 实现
├── graph/                 # 计算图模块实现
│   ├── CMakeLists.txt
│   ├── node.cpp          # 节点实现
│   └── graph.cpp         # 图实现
├── runtime/               # 运行时模块实现
│   ├── CMakeLists.txt
│   └── engine.cpp        # 引擎实现
└── utils/                 # 工具模块实现
    ├── CMakeLists.txt
    └── logger.cpp        # 日志实现
```

## tests/ - 测试目录

单元测试和集成测试：

```
tests/
├── CMakeLists.txt         # 测试构建配置
├── test_tensor.cpp        # 张量测试
├── test_backend.cpp       # 后端测试
└── test_graph.cpp         # 计算图测试
```

## examples/ - 示例目录

使用示例代码：

```
examples/
├── CMakeLists.txt         # 示例构建配置
├── simple_inference.cpp   # 简单推理示例
└── build_graph.cpp        # 构建计算图示例
```

## docs/ - 文档目录

项目文档：

```
docs/
├── API.md                 # API 参考文档
├── ARCHITECTURE.md        # 架构设计文档
├── BUILD.md              # 构建指南
└── GETTING_STARTED.md    # 快速入门指南
```

## 模块依赖关系

```
依赖图（从上到下，上层依赖下层）：

┌─────────────────┐
│    examples     │  示例应用
└────────┬────────┘
         │
┌────────▼────────┐
│     runtime     │  运行时引擎
└────────┬────────┘
         │
    ┌────▼────┐
    │  graph  │     计算图
    └────┬────┘
         │
  ┌──────▼──────┐
  │  operators  │   算子层
  └──────┬──────┘
         │
   ┌─────▼─────┐
   │  backends │    后端抽象
   └─────┬─────┘
         │
    ┌────▼────┐
    │  core   │      核心层
    └─────────┘
         │
    ┌────▼────┐
    │  utils  │      工具层
    └─────────┘
```

## 构建产物

构建后的目录结构：

```
build/
├── CMakeCache.txt
├── CMakeFiles/
├── bin/                   # 可执行文件
│   ├── Debug/            # Debug 构建
│   │   ├── test_tensor.exe
│   │   ├── test_backend.exe
│   │   ├── test_graph.exe
│   │   ├── simple_inference.exe
│   │   └── build_graph.exe
│   └── Release/          # Release 构建
│       └── ...
└── lib/                   # 库文件
    ├── Debug/
    │   ├── mini_infer_core.lib
    │   ├── mini_infer_backends.lib
    │   ├── mini_infer_operators.lib
    │   ├── mini_infer_graph.lib
    │   ├── mini_infer_runtime.lib
    │   └── mini_infer_utils.lib
    └── Release/
        └── ...
```

## 关键文件说明

### 配置文件

- **CMakeLists.txt**: 主构建配置，定义编译选项、依赖关系、安装规则
- **CMakePresets.json**: CMake 预设配置，提供常用构建配置
- **.clang-format**: 代码格式化规则
- **.gitignore**: Git 版本控制忽略规则

### 构建脚本

- **build.ps1**: Windows PowerShell 构建脚本
  - 支持 Debug/Release 构建
  - 支持清理、测试、安装
  - 多处理器并行编译

- **build.sh**: Linux/macOS Shell 构建脚本
  - 功能与 PowerShell 脚本对应
  - 支持命令行参数配置

### 文档文件

- **README.md**: 项目介绍和快速开始
- **LICENSE**: MIT 开源许可证
- **CONTRIBUTING.md**: 贡献者指南
- **PROJECT_STRUCTURE.md**: 项目结构说明（本文件）

## 模块说明

### Core 模块
- **职责**: 提供基础数据结构（Tensor, Shape, Allocator）
- **输出**: libmini_infer_core
- **依赖**: 仅依赖标准库

### Backends 模块
- **职责**: 硬件抽象层，支持不同计算后端
- **输出**: libmini_infer_backends
- **依赖**: mini_infer_core

### Operators 模块
- **职责**: 实现深度学习算子
- **输出**: libmini_infer_operators
- **依赖**: mini_infer_core, mini_infer_backends

### Graph 模块
- **职责**: 计算图表示和管理
- **输出**: libmini_infer_graph
- **依赖**: mini_infer_core, mini_infer_operators

### Runtime 模块
- **职责**: 推理引擎，执行计算图
- **输出**: libmini_infer_runtime
- **依赖**: mini_infer_core, mini_infer_backends, mini_infer_operators, mini_infer_graph, mini_infer_utils

### Utils 模块
- **职责**: 日志、性能分析等工具
- **输出**: libmini_infer_utils
- **依赖**: 标准库

## 命名约定

### 文件命名
- 头文件: `lowercase_with_underscores.h`
- 源文件: `lowercase_with_underscores.cpp`
- CMake 文件: `CMakeLists.txt`

### 代码命名
- 类名: `PascalCase` (如 `TensorShape`)
- 函数名: `snake_case` (如 `allocate_memory`)
- 变量名: `snake_case` (如 `input_tensor`)
- 常量: `UPPER_CASE` (如 `MAX_BATCH_SIZE`)
- 命名空间: `snake_case` (如 `mini_infer::core`)

### 库命名
- 格式: `libmini_infer_<module>.a` (静态库)
- 格式: `libmini_infer_<module>.so` (动态库，Linux)
- 格式: `mini_infer_<module>.lib` (静态库，Windows)
- 格式: `mini_infer_<module>.dll` (动态库，Windows)

## 扩展指南

### 添加新模块

1. 在 `include/mini_infer/` 创建头文件目录
2. 在 `src/` 创建对应的实现目录
3. 添加 `CMakeLists.txt` 配置
4. 在顶层 `CMakeLists.txt` 中添加 `add_subdirectory()`
5. 在 `include/mini_infer/mini_infer.h` 中包含新头文件

### 添加新算子

1. 在 `include/mini_infer/operators/` 创建头文件
2. 在 `src/operators/` 创建实现文件
3. 更新 `src/operators/CMakeLists.txt`
4. 添加测试文件到 `tests/`
5. 更新文档

### 添加新后端

1. 在 `include/mini_infer/backends/` 创建头文件
2. 在 `src/backends/` 创建实现文件
3. 更新 `BackendFactory`
4. 添加测试
5. 更新构建配置（如果需要新的依赖）

## 维护建议

1. **保持模块独立性**: 避免循环依赖
2. **文档同步更新**: 代码变更时更新相应文档
3. **测试覆盖**: 新功能需要添加测试
4. **代码审查**: 使用 Pull Request 机制
5. **版本管理**: 遵循语义化版本规范

## 工具支持

### IDE 配置
- **VS Code**: 使用 CMake Tools 插件
- **CLion**: 直接打开 CMakeLists.txt
- **Visual Studio**: 打开文件夹或使用 CMake 项目

### 调试
- Debug 构建包含调试符号
- 使用 GDB (Linux) 或 LLDB (macOS) 或 Visual Studio Debugger (Windows)

### 性能分析
- 使用 `valgrind` (Linux)
- 使用 `perf` (Linux)
- 使用 Visual Studio Profiler (Windows)

## 版本历史

- **v0.1.0** (当前): 初始版本，基础框架搭建
  - 核心数据结构
  - CPU 后端
  - 基础算子框架
  - 计算图系统
  - 推理引擎

## 未来规划

### 短期 (v0.2.0)
- 完善常用算子实现
- 添加更多测试用例
- 性能优化

### 中期 (v0.3.0)
- CUDA 后端支持
- 图优化 pass
- ONNX 模型加载

### 长期 (v1.0.0)
- 量化支持 (INT8)
- 自动调优
- 模型压缩

---

最后更新: 2025-11-07

