# Conan 构建指南

本文档介绍如何使用 Conan 包管理器构建 Mini-Infer 项目。

> **📌 相关文档:**
> - [CUDA 配置指南](CUDA_CONAN_SETUP.md) - 如何启用 CUDA 支持
> - [快速开始](../QUICK_START.md) - 快速上手指南
> - [构建指南](BUILD.md) - 详细的构建说明

## 为什么选择 Conan？

- ✅ **真正的跨平台**: 一套命令在 Windows/Linux/macOS 上都能工作
- ✅ **自动依赖管理**: 自动下载、编译和配置所有依赖（如 Protobuf）
- ✅ **可重现构建**: 锁定依赖版本，确保构建一致性
- ✅ **与 CMake 完美集成**: 自动生成 CMakePresets.json 和工具链文件
- ✅ **灵活的选项系统**: 通过 Conan 选项控制功能开关（ONNX/CUDA/Logging）

## 前置要求

### 安装 Conan 2.x

```bash
# 使用 pip 安装
pip install conan

# 验证安装
conan --version
```

### 配置 Conan Profile

首次使用时，Conan 会自动检测系统配置：

```bash
conan profile detect --force
```

## 快速开始

### Windows (MSVC)

```powershell
# 1. 安装依赖并生成 CMake 工具链
conan install . --output-folder=build/windows-conan-debug --build=missing -s build_type=Debug

# 2. 配置项目（使用 Conan 生成的工具链）
cmake --preset windows-conan-debug

# 3. 编译
cmake --build build/windows-conan-debug

# 4. 运行示例
.\build\windows-conan-debug\bin\onnx_parser_example.exe .\models\python\lenet5\models\lenet5.onnx
```

### Linux (GCC/Clang)

```bash
# 1. 安装依赖
conan install . --output-folder=build/linux-conan-debug --build=missing -s build_type=Debug

# 2. 配置项目
cmake --preset linux-conan-debug

# 3. 编译
cmake --build build/linux-conan-debug

# 4. 运行示例
./build/linux-conan-debug/bin/onnx_parser_example ./models/python/lenet5/models/lenet5.onnx
```

## 构建选项

### Conan 选项

在 `conan install` 时可以自定义选项，**这些选项会自动传递到生成的 CMakePresets.json 中**：

```bash
# 启用/禁用 ONNX 支持
conan install . \
  --output-folder=build/xxx \
  -o enable_onnx=True \      # 默认 ON → 生成 MINI_INFER_ENABLE_ONNX=ON
  --build=missing

# 禁用所有可选功能
conan install . \
  --output-folder=build/xxx \
  -o enable_onnx=False \     # → MINI_INFER_ENABLE_ONNX=OFF
  -o enable_logging=False \  # → MINI_INFER_ENABLE_LOGGING=OFF
  --build=missing

# 启用 CUDA（未来支持）
conan install . \
  --output-folder=build/xxx \
  -o enable_cuda=True \      # → MINI_INFER_ENABLE_CUDA=ON
  --build=missing
```

**自动传递原理：**
- Conan 的 `generate()` 方法会将选项转换为 CMake 缓存变量
- 生成的 `CMakePresets.json` 会包含这些变量
- 无需在项目的 `CMakePresets.json` 中手动配置

### 构建类型

```bash
# Debug 构建
conan install . -s build_type=Debug

# Release 构建
conan install . -s build_type=Release
```

## 完整工作流示例

### Release 构建（Windows）

```powershell
# 安装依赖（Release 模式）
conan install . `
  --output-folder=build/windows-conan-release `
  --build=missing `
  -s build_type=Release `
  -o enable_onnx=True `
  -o enable_logging=True

# 配置并编译
cmake --preset windows-conan-release
cmake --build build/windows-conan-release --config Release

# 安装（可选）
cmake --install build/windows-conan-release --prefix install
```

### Release 构建（Linux）

```bash
# 安装依赖
conan install . \
  --output-folder=build/linux-conan-release \
  --build=missing \
  -s build_type=Release \
  -o enable_onnx=True

# 配置并编译
cmake --preset linux-conan-release
cmake --build build/linux-conan-release

# 安装
cmake --install build/linux-conan-release --prefix install
```

## 依赖说明

### 当前依赖

- **Protobuf 3.21.12**: ONNX 模型解析所需（当 `enable_onnx=True` 时）

### 依赖版本锁定

Conan 会自动处理依赖的传递依赖（如 Protobuf 的 Abseil 依赖），无需手动配置。

## 清理构建

```bash
# 清理 Conan 缓存
conan remove "*" -c

# 清理构建目录
rm -rf build/
```

## 故障排除

### 问题: Protobuf 找不到

**解决方案**: 确保在运行 CMake 配置前先运行 `conan install`:

```bash
# 正确的顺序
conan install . --output-folder=build/xxx --build=missing
cmake --preset xxx
```

### 问题: 工具链文件找不到

**解决方案**: 检查 `conan install` 的输出目录是否与 CMakePresets.json 中的一致。

### 问题: 编译器不兼容

**解决方案**: 使用 `conan profile` 检查并调整编译器配置:

```bash
# 查看当前 profile
conan profile show default

# 手动编辑 profile
conan profile path default
```

## 高级用法

### 使用 Ninja 生成器（推荐，构建更快）

Ninja 是一个快速的构建系统，比传统的 Make/MSBuild 更快：

```bash
# 使用 Ninja 生成器 + C++20 标准
conan install . \
  -s build_type=Release \
  -s compiler.cppstd=20 \
  -c tools.cmake.cmaketoolchain:generator=Ninja \
  --build missing

# 配置和构建（会使用 Ninja）
cmake --preset windows-conan-release  # 或 linux-conan-release
cmake --build build/windows-conan-release
```

**说明:**
- `-s compiler.cppstd=20`: 指定 C++ 标准（17/20/23）
- `-c tools.cmake.cmaketoolchain:generator=Ninja`: 使用 Ninja 生成器
- `--build missing`: 如果二进制包不存在，则从源码编译

**性能对比:**
- Make: ~60 秒（8核）
- MSBuild: ~45 秒（8核）
- Ninja: ~25 秒（8核）⚡

**注意:** 需要先安装 Ninja:
```bash
# Windows (使用 scoop)
scoop install ninja

# Linux (Ubuntu/Debian)
sudo apt-get install ninja-build

# macOS
brew install ninja
```

### 自定义 Profile

创建 `conanprofile.txt`:

```ini
[settings]
os=Windows
arch=x86_64
compiler=msvc
compiler.version=193
compiler.runtime=dynamic
build_type=Release

[options]
mini-infer:enable_onnx=True
mini-infer:enable_logging=True

[conf]
tools.cmake.cmaketoolchain:generator=Ninja
```

使用自定义 profile:

```bash
conan install . --profile=conanprofile.txt --build=missing
```

### 锁定依赖版本

生成 lockfile:

```bash
conan lock create . --lockfile=conan.lock
```

使用 lockfile 构建:

```bash
conan install . --lockfile=conan.lock --build=missing
```


## CUDA 支持

如需启用 CUDA 支持，请参考 [CUDA 配置指南](CUDA_CONAN_SETUP.md)。

简要示例：

```bash
# Windows
conan install . --output-folder=build --build=missing \
  -o enable_cuda=True \
  -o cuda_toolkit_root="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3"

# Linux
conan install . --output-folder=build --build=missing \
  -o enable_cuda=True \
  -o cuda_toolkit_root="/usr/local/cuda"

cmake --preset conan-release
cmake --build --preset conan-release
```

## 参考资料

- [Conan 官方文档](https://docs.conan.io/)
- [Conan CMake 集成](https://docs.conan.io/2/reference/tools/cmake.html)
- [CMakePresets.json 规范](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html)
- [Mini-Infer CUDA 配置](CUDA_CONAN_SETUP.md)
