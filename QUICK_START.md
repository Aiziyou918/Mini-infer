# Mini-Infer 快速开始指南

## 🚀 一分钟快速开始

### Windows
```powershell
# 1. 安装 Conan
pip install conan

# 2. 初始化
conan profile detect --force

# 3. 一键构建（使用自动化脚本）
.\build.ps1

# 4. 运行示例
.\build\Debug\bin\onnx_parser_example.exe .\models\python\lenet5\models\lenet5.onnx
```

### Linux/macOS
```bash
# 1. 安装 Conan
pip install conan

# 2. 初始化
conan profile detect --force

# 3. 一键构建（使用自动化脚本）
chmod +x build.sh
./build.sh

# 4. 运行示例
./build/Debug/bin/onnx_parser_example ./models/python/lenet5/models/lenet5.onnx
```

## 📋 手动构建流程

### 基本流程（3 步）

```bash
# 步骤 1: 安装依赖（Conan 会自动生成 CMake 预设）
conan install . -s build_type=Debug -o enable_onnx=True --build=missing

# 步骤 2: 配置 CMake（使用 Conan 生成的预设）
cmake --preset conan-debug

# 步骤 3: 编译
cmake --build build/Debug
```

### Release 构建

```bash
conan install . -s build_type=Release -o enable_onnx=True --build=missing
cmake --preset conan-release
cmake --build build/Release
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
-o enable_cuda=True   # 启用 CUDA GPU 加速（未来支持）
-o enable_cuda=False  # 仅 CPU 模式
```

### 示例：自定义配置

```bash
# 最小化构建（无 ONNX，无日志）
conan install . -s build_type=Release -o enable_onnx=False -o enable_logging=False --build=missing

# 完整功能（ONNX + 日志）
conan install . -s build_type=Debug -o enable_onnx=True -o enable_logging=True --build=missing
```

## 🔧 自动化脚本

### Windows (PowerShell)

```powershell
# 基本用法
.\build.ps1                          # Debug 构建
.\build.ps1 -BuildType Release       # Release 构建
.\build.ps1 -Clean                   # 清理并构建
.\build.ps1 -Test                    # 构建并运行测试
.\build.ps1 -Install                 # 构建并安装

# 组合使用
.\build.ps1 -BuildType Release -Clean -Test -Install
```

### Linux/macOS (Bash)

```bash
# 基本用法
./build.sh                    # Debug 构建
./build.sh -r                 # Release 构建
./build.sh -c                 # 清理并构建
./build.sh -t                 # 构建并运行测试
./build.sh -i                 # 构建并安装

# 禁用功能
./build.sh --no-onnx          # 禁用 ONNX
./build.sh --no-logging       # 禁用日志

# 组合使用
./build.sh -r -c -t -i        # Release + 清理 + 测试 + 安装
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
│   │   ├── lib/                    # 库文件
│   │   └── generators/             # Conan 生成的文件
│   │       ├── conan_toolchain.cmake
│   │       └── CMakePresets.json   # 自动生成的预设
│   └── Release/                    # Release 构建目录
│       └── ...
├── third_party/onnx/              # ONNX proto 文件（自动生成）
│   ├── onnx.proto
│   ├── onnx.pb.h
│   └── onnx.pb.cc
└── install/                       # 安装目录（可选）
```

## 🧪 运行测试

```bash
# 使用 Conan 生成的预设
ctest --preset conan-debug
ctest --preset conan-release

# 或在构建目录中运行
cd build/Debug
ctest --output-on-failure
```

## 🎯 常见任务

### 清理重新构建

```bash
# 删除构建目录
rm -rf build/

# 重新构建
conan install . -s build_type=Debug -o enable_onnx=True --build=missing
cmake --preset conan-debug
cmake --build build/Debug
```

### 只重新配置 CMake

```bash
# 不需要重新运行 conan install，只重新配置
cmake --preset conan-debug
```

### 只重新编译

```bash
# 不重新配置，只编译
cmake --build build/Debug
```

### 查看可用预设

```bash
# 查看 Conan 生成了哪些预设
cmake --list-presets
```

## ❓ 常见问题

### Q: 找不到 protoc？
**A**: 使用 Conan 后不需要手动安装 Protobuf。Conan 会自动下载并配置。

### Q: ONNX 支持被禁用？
**A**: 确保使用了 `-o enable_onnx=True` 选项：
```bash
conan install . -o enable_onnx=True --build=missing
```

### Q: 如何禁用 ONNX？
**A**: 使用 `-o enable_onnx=False`：
```bash
conan install . -o enable_onnx=False --build=missing
```

### Q: 编译速度慢？
**A**: 自动化脚本 (`build.ps1`/`build.sh`) 会自动检测并建议安装 Ninja 生成器，可以提升 50%+ 的编译速度。如果你手动构建，可以这样使用 Ninja：
```bash
conan install . -c tools.cmake.cmaketoolchain:generator=Ninja --build=missing
```

### Q: 脚本检测到没有 Ninja 怎么办？
**A**: 脚本会询问是否安装，你可以：
- 输入 `Y` 并按提示安装 Ninja，然后继续
- 输入 `n` 跳过，使用默认生成器（Visual Studio 或 Unix Makefiles）继续构建

## 📚 更多文档

- **[完整 README](README.md)** - 项目概述和详细说明
- **[Conan 构建指南](docs/CONAN_BUILD_GUIDE.md)** - Conan 详细使用说明
- **[ONNX 解析器设计](docs/ONNX_PARSER_DESIGN.md)** - ONNX 解析器架构文档
- **[Conan 选项指南](docs/CONAN_OPTIONS_GUIDE.md)** - Conan 选项详细说明

## 💡 提示

- ✅ 优先使用自动化脚本 (`build.ps1`/`build.sh`)
- ✅ Conan 选项会自动传递到 CMake
- ✅ 不需要手动指定 `--output-folder`，使用默认即可
- ✅ CMake 预设由 Conan 自动生成，无需手动创建
- ✅ 第一次构建会下载依赖，后续构建很快
