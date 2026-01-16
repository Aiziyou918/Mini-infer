# Mini-Infer 快速开始指南

## 🚀 一分钟快速开始

### Windows
```powershell
# 1. 安装 Conan
pip install conan

# 2. 初始化
conan profile detect --force

# 3. 安装依赖并生成 CMake 预设
conan install . --output-folder=build --build=missing -s build_type=Debug

# 4. 配置并编译
cmake --preset conan-debug
cmake --build --preset conan-debug

# 5. 运行示例
.\build\Debug\bin\onnx_parser_example.exe .\models\python\lenet5\models\lenet5.onnx
```

### Linux/macOS
```bash
# 1. 安装 Conan
pip install conan

# 2. 初始化
conan profile detect --force

# 3. 安装依赖并生成 CMake 预设
conan install . --output-folder=build --build=missing -s build_type=Debug

# 4. 配置并编译
cmake --preset conan-debug
cmake --build --preset conan-debug

# 5. 运行示例
./build/Debug/bin/onnx_parser_example ./models/python/lenet5/models/lenet5.onnx
```

Tiny-BERT 情感分类示例：

```bash
# Windows
.\build\Debug\bin\bert_inference.exe ^
  .\models\python\bert\models\bert_tiny.onnx ^
  .\models\python\bert\test_samples ^
  10

# Linux/macOS
./build/Debug/bin/bert_inference \
  ./models/python/bert/models/bert_tiny.onnx \
  ./models/python/bert/test_samples \
  10
```

## 📋 详细构建流程

### 基本流程（3 步）

```bash
# 步骤 1: 安装依赖（Conan 会自动生成 CMake 预设）
conan install . --output-folder=build --build=missing -s build_type=Debug

# 步骤 2: 配置 CMake（使用 Conan 生成的预设）
cmake --preset conan-debug

# 步骤 3: 编译
cmake --build --preset conan-debug
```

### Release 构建

```bash
# 步骤 1: 安装依赖
conan install . --output-folder=build --build=missing -s build_type=Release

# 步骤 2: 配置
cmake --preset conan-release

# 步骤 3: 编译
cmake --build --preset conan-release
```

## 🎛️ 构建选项

### Conan 选项

```bash
# 启用/禁用 ONNX 支持（默认：启用）
-o enable_onnx=True   # 启用 ONNX 模型导入
-o enable_onnx=False  # 禁用（不会安装 Protobuf）

# 启用/禁用日志（默认：启用）
-o enable_logging=True   # 启用日志输出
-o enable_logging=False  # 禁用日志（性能优化）

# 启用/禁用 CUDA（默认：禁用）
-o enable_cuda=True   # 启用 CUDA GPU 加速
-o enable_cuda=False  # 仅 CPU 模式

# 指定 CUDA 路径（启用 CUDA 时可选）
-o cuda_toolkit_root="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3"
```

### 示例：自定义配置

```bash
# 最小化构建（无 ONNX，无日志）
conan install . --output-folder=build --build=missing \
  -s build_type=Release \
  -o enable_onnx=False \
  -o enable_logging=False

# 完整功能（ONNX + 日志）
conan install . --output-folder=build --build=missing \
  -s build_type=Debug \
  -o enable_onnx=True \
  -o enable_logging=True

# 启用 CUDA 支持
conan install . --output-folder=build --build=missing \
  -s build_type=Release \
  -o enable_cuda=True \
  -o cuda_toolkit_root="/usr/local/cuda"
```

## 🔧 使用 CMake Presets

Conan 会自动生成 `CMakePresets.json`，你可以直接使用这些预设：

```bash
# 查看可用的预设
cmake --list-presets

# 使用预设配置
cmake --preset conan-debug      # Debug 配置
cmake --preset conan-release    # Release 配置

# 使用预设构建
cmake --build --preset conan-debug
cmake --build --preset conan-release

# 使用预设测试
ctest --preset conan-debug
ctest --preset conan-release
```

## 📁 目录结构

构建后的目录结构：

```
Mini-Infer/
├── build/
│   ├── Debug/                      # Debug 构建目录
│   │   ├── bin/                    # 可执行文件
│   │   │   ├── onnx_parser_example.exe
│   │   │   └── ...
│   │   └── lib/                    # 库文件
│   ├── Release/                    # Release 构建目录
│   │   └── ...
│   ├── generators/                 # Conan 生成的文件
│   │   ├── conan_toolchain.cmake
│   │   ├── CMakePresets.json       # 自动生成的预设
│   │   ├── CMakeDeps.cmake
│   │   └── ...
│   └── CMakeUserPresets.json       # 用户自定义预设（可选）
├── third_party/onnx/              # ONNX proto 文件（自动生成）
│   ├── onnx.proto
│   ├── onnx.pb.h
│   └── onnx.pb.cc
└── conanfile.py                   # Conan 配置文件
```

## 🧪 运行测试

```bash
# 使用 Conan 生成的预设
ctest --preset conan-debug
ctest --preset conan-release

# 或在构建目录中运行
cd build/Debug
ctest --output-on-failure

# 并行运行测试
ctest -j8 --output-on-failure
```

## 🎯 常见任务

### 清理重新构建

```bash
# 删除构建目录
rm -rf build/

# 重新构建
conan install . --output-folder=build --build=missing -s build_type=Debug
cmake --preset conan-debug
cmake --build --preset conan-debug
```

### 只重新配置 CMake

```bash
# 不需要重新运行 conan install，只重新配置
cmake --preset conan-debug
```

### 只重新编译

```bash
# 不重新配置，只编译
cmake --build --preset conan-debug
```

### 增量编译（修改代码后）

```bash
# 直接编译，CMake 会自动检测变化
cmake --build --preset conan-debug

# 或指定并行任务数
cmake --build --preset conan-debug -j8
```

### 查看可用预设

```bash
# 查看 Conan 生成了哪些预设
cmake --list-presets

# 查看构建预设
cmake --list-presets=build

# 查看测试预设
cmake --list-presets=test
```

## ❓ 常见问题

### Q: 找不到 protoc？
**A**: 使用 Conan 后不需要手动安装 Protobuf。Conan 会自动下载并配置。

### Q: ONNX 支持被禁用？
**A**: 确保使用了 `-o enable_onnx=True` 选项（这是默认值）：
```bash
conan install . --output-folder=build --build=missing -o enable_onnx=True
```

### Q: 如何禁用 ONNX？
**A**: 使用 `-o enable_onnx=False`：
```bash
conan install . --output-folder=build --build=missing -o enable_onnx=False
```

### Q: 编译速度慢？
**A**: 可以使用 Ninja 生成器来提升编译速度：
```bash
# 安装 Ninja
pip install ninja  # 或 apt-get install ninja-build

# 使用 Ninja 生成器
conan install . --output-folder=build --build=missing \
  -c tools.cmake.cmaketoolchain:generator=Ninja

cmake --preset conan-debug
cmake --build --preset conan-debug -j8
```

### Q: Conan 找不到依赖？
**A**: 首次使用需要检测 profile：
```bash
conan profile detect --force
```

### Q: 如何清理 Conan 缓存？
**A**: 如果遇到依赖问题，可以清理缓存：
```bash
# 清理所有缓存
conan remove "*" -c

# 清理特定包
conan remove "protobuf/*" -c
```

### Q: CMake 找不到预设？
**A**: 确保先运行了 `conan install`：
```bash
# 步骤 1: 先安装依赖（生成预设）
conan install . --output-folder=build --build=missing

# 步骤 2: 然后才能使用预设
cmake --preset conan-debug
```

## 📚 更多文档

- **[完整 README](README.md)** - 项目概述和详细说明
- **[Conan 构建指南](docs/CONAN_BUILD_GUIDE.md)** - Conan 详细使用说明
- **[CUDA 配置指南](docs/CUDA_CONAN_SETUP.md)** - CUDA 后端配置
- **[入门教程](docs/GETTING_STARTED.md)** - 完整的入门教程
- **[架构设计](docs/ARCHITECTURE.md)** - 架构设计文档
- **[API 文档](docs/API.md)** - API 参考手册

## 💡 提示

- ✅ 使用 `--output-folder=build` 统一输出目录
- ✅ Conan 选项会自动传递到 CMake
- ✅ CMake 预设由 Conan 自动生成，无需手动创建
- ✅ 第一次构建会下载依赖，后续构建很快
- ✅ 使用 `cmake --build --preset <preset> -j8` 并行编译
- ✅ 修改代码后只需运行 `cmake --build --preset <preset>` 增量编译
