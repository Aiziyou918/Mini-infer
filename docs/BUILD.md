# Mini-Infer 构建指南

本文档详细说明如何使用 Conan 包管理器构建 Mini-Infer 项目。

---

## 系统要求

### 通用要求
- **CMake** 3.18 或更高版本
- **Conan** 2.0 或更高版本
- **Python** 3.7+ (用于安装 Conan)
- **C++17** 兼容的编译器

### Windows
- Visual Studio 2017 或更高版本（推荐 VS 2022）
- PowerShell 5.0+

### Linux
- GCC 7+ 或 Clang 5+
- Make 或 Ninja

### macOS
- Xcode 10+
- Command Line Tools

---

## 安装 Conan

### 使用 pip 安装（推荐）

```bash
# 安装 Conan
pip install conan

# 验证安装
conan --version

# 检测默认配置
conan profile detect --force
```

### 使用系统包管理器

```bash
# Ubuntu/Debian
sudo apt-get install python3-pip
pip3 install conan

# macOS (Homebrew)
brew install conan

# Arch Linux
sudo pacman -S conan
```

---

## 快速构建

### 三步构建流程

```bash
# 步骤 1: 安装依赖并生成 CMake 预设
conan install . --output-folder=build --build=missing -s build_type=Release

# 步骤 2: 配置 CMake
cmake --preset conan-release

# 步骤 3: 编译
cmake --build --preset conan-release
```

### Debug 构建

```bash
# 步骤 1: 安装依赖
conan install . --output-folder=build --build=missing -s build_type=Debug

# 步骤 2: 配置
cmake --preset conan-debug

# 步骤 3: 编译
cmake --build --preset conan-debug
```

---

## 构建选项

### Conan 选项

Mini-Infer 提供以下 Conan 选项：

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_onnx` | bool | `True` | 启用 ONNX 模型导入支持 |
| `enable_logging` | bool | `True` | 启用日志输出 |
| `enable_cuda` | bool | `False` | 启用 CUDA GPU 加速 |
| `cuda_toolkit_root` | string | `""` | CUDA Toolkit 安装路径 |

### 使用示例

```bash
# 完整功能构建（默认）
conan install . --output-folder=build --build=missing \
  -s build_type=Release \
  -o enable_onnx=True \
  -o enable_logging=True

# 最小化构建（无 ONNX，无日志）
conan install . --output-folder=build --build=missing \
  -s build_type=Release \
  -o enable_onnx=False \
  -o enable_logging=False

# 启用 CUDA 支持
conan install . --output-folder=build --build=missing \
  -s build_type=Release \
  -o enable_cuda=True \
  -o cuda_toolkit_root="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3"
```

---

## CMake 配置选项

除了 Conan 选项外，还可以通过 CMake 变量进行配置：

```bash
# 是否构建测试（默认 ON）
-DMINI_INFER_BUILD_TESTS=ON|OFF

# 是否构建示例（默认 ON）
-DMINI_INFER_BUILD_EXAMPLES=ON|OFF

# 是否构建共享库（默认 ON）
-DMINI_INFER_BUILD_SHARED_LIBS=ON|OFF

# 是否启用性能分析（默认 ON）
-DMINI_INFER_ENABLE_PROFILING=ON|OFF
```

这些选项会在 `cmake --preset` 时自动从 Conan 传递。

---

## 使用 CMake Presets

Conan 会自动生成 `build/generators/CMakePresets.json`，包含以下预设：

### 查看可用预设

```bash
# 查看所有预设
cmake --list-presets

# 输出示例：
# Available configure presets:
#   "conan-debug"    - Debug build
#   "conan-release"  - Release build
```

### 使用预设

```bash
# 配置
cmake --preset conan-release

# 构建
cmake --build --preset conan-release

# 测试
ctest --preset conan-release

# 安装
cmake --build --preset conan-release --target install
```

---

## 高级构建配置

### 使用 Ninja 生成器（提升编译速度）

```bash
# 安装 Ninja
pip install ninja

# 配置 Conan 使用 Ninja
conan install . --output-folder=build --build=missing \
  -c tools.cmake.cmaketoolchain:generator=Ninja

# 构建（Ninja 会自动并行）
cmake --preset conan-release
cmake --build --preset conan-release
```

### 并行编译

```bash
# 使用多核编译（8 个并行任务）
cmake --build --preset conan-release -j8

# 使用所有可用核心
cmake --build --preset conan-release -j$(nproc)  # Linux
cmake --build --preset conan-release -j$env:NUMBER_OF_PROCESSORS  # Windows PowerShell
```

### 指定编译器

```bash
# 使用特定编译器
conan install . --output-folder=build --build=missing \
  -s compiler=gcc \
  -s compiler.version=11

# 或通过环境变量
export CC=gcc-11
export CXX=g++-11
conan install . --output-folder=build --build=missing
```

---

## 运行测试

### 使用 CTest

```bash
# 运行所有测试
ctest --preset conan-debug

# 详细输出
ctest --preset conan-debug --output-on-failure

# 并行运行测试
ctest --preset conan-debug -j8

# 运行特定测试
ctest --preset conan-debug -R test_tensor
```

### 直接运行测试可执行文件

```bash
# Windows
.\build\Debug\bin\test_tensor.exe
.\build\Debug\bin\test_backend.exe

# Linux/macOS
./build/Debug/bin/test_tensor
./build/Debug/bin/test_backend
```

---

## 运行示例

```bash
# Windows
.\build\Release\bin\onnx_parser_example.exe .\models\python\lenet5\models\lenet5.onnx

# Linux/macOS
./build/Release/bin/onnx_parser_example ./models/python/lenet5/models/lenet5.onnx
```

---

## 安装

### 安装到默认位置

```bash
# 使用预设安装
cmake --build --preset conan-release --target install

# 或手动指定
cd build/Release
cmake --install .
```

### 安装到自定义位置

```bash
# 配置时指定安装前缀
cmake --preset conan-release -DCMAKE_INSTALL_PREFIX=/path/to/install

# 构建并安装
cmake --build --preset conan-release
cmake --install build/Release --prefix /path/to/install
```

### 安装后的目录结构

```
install/
├── include/
│   └── mini_infer/
│       ├── core/
│       ├── backends/
│       ├── operators/
│       ├── graph/
│       ├── runtime/
│       ├── importers/
│       └── utils/
├── lib/
│   ├── libmini_infer_core.a
│   ├── libmini_infer_backends.a
│   ├── libmini_infer_operators.a
│   ├── libmini_infer_graph.a
│   ├── libmini_infer_runtime.a
│   ├── libmini_infer_importers.a
│   └── libmini_infer_utils.a
└── lib/cmake/Mini-Infer/
    ├── Mini-InferConfig.cmake
    └── Mini-InferTargets.cmake
```

---

## 在自己的项目中使用

### 方式 1: 使用 find_package

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyProject)

# 查找 Mini-Infer
find_package(Mini-Infer REQUIRED)

# 链接库
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE 
    MiniInfer::mini_infer_runtime
    MiniInfer::mini_infer_importers
)
```

### 方式 2: 使用 add_subdirectory

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyProject)

# 添加 Mini-Infer 子目录
add_subdirectory(path/to/Mini-Infer)

# 链接库
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE 
    mini_infer_runtime
    mini_infer_importers
)
```

### 方式 3: 使用 Conan 依赖

创建 `conanfile.txt`:

```ini
[requires]
mini-infer/0.1.0

[generators]
CMakeDeps
CMakeToolchain

[options]
mini-infer/*:enable_onnx=True
```

然后在 CMakeLists.txt 中：

```cmake
find_package(mini-infer REQUIRED)
target_link_libraries(my_app PRIVATE mini-infer::mini-infer)
```

---

## 常见问题

### 问题 1: Conan 找不到 profile

**症状:**
```
ERROR: Conan profile not found
```

**解决方案:**
```bash
conan profile detect --force
```

### 问题 2: 依赖下载失败

**症状:**
```
ERROR: Failed to download protobuf/3.21.12
```

**解决方案:**
```bash
# 清理缓存重试
conan remove "*" -c
conan install . --output-folder=build --build=missing
```

### 问题 3: CMake 找不到预设

**症状:**
```
CMake Error: No such preset in CMakePresets.json: 'conan-release'
```

**解决方案:**
确保先运行了 `conan install`：
```bash
# 步骤 1: 生成预设
conan install . --output-folder=build --build=missing

# 步骤 2: 使用预设
cmake --preset conan-release
```

### 问题 4: 编译器版本不兼容

**症状:**
```
ERROR: Compiler version not supported
```

**解决方案:**
更新 Conan profile：
```bash
# 编辑 profile
conan profile show default

# 或创建新 profile
conan profile detect --force
```

### 问题 5: 链接错误

**症状:**
```
undefined reference to ...
```

**解决方案:**
确保链接顺序正确：
```cmake
target_link_libraries(your_target PRIVATE
    mini_infer_runtime      # 最上层
    mini_infer_importers
    mini_infer_graph
    mini_infer_operators
    mini_infer_backends
    mini_infer_core         # 最底层
    mini_infer_utils
)
```

---

## 性能优化

### Release 构建优化

```bash
# 使用 Release 模式
conan install . --output-folder=build --build=missing -s build_type=Release

# 禁用日志（生产环境）
conan install . --output-folder=build --build=missing \
  -s build_type=Release \
  -o enable_logging=False
```

### 编译器优化标志

Conan 会自动根据 build_type 设置优化标志：

- **Debug**: `-g -O0`
- **Release**: `-O3 -DNDEBUG`
- **RelWithDebInfo**: `-O2 -g -DNDEBUG`
- **MinSizeRel**: `-Os -DNDEBUG`

如需自定义：

```bash
# GCC/Clang
conan install . --output-folder=build --build=missing \
  -c tools.cmake.cmaketoolchain:extra_cxxflags="-march=native -mtune=native"

# MSVC
conan install . --output-folder=build --build=missing \
  -c tools.cmake.cmaketoolchain:extra_cxxflags="/O2 /Ob2 /GL"
```

---

## 清理

### 清理构建目录

```bash
# 删除整个构建目录
rm -rf build/

# Windows PowerShell
Remove-Item -Recurse -Force build\
```

### 清理 Conan 缓存

```bash
# 清理所有缓存
conan remove "*" -c

# 清理特定包
conan remove "protobuf/*" -c
```

---

## 持续集成

### GitHub Actions 示例

```yaml
name: Build

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Conan
      run: pip install conan
    
    - name: Detect Conan profile
      run: conan profile detect --force
    
    - name: Install dependencies
      run: conan install . --output-folder=build --build=missing -s build_type=Release
    
    - name: Configure
      run: cmake --preset conan-release
    
    - name: Build
      run: cmake --build --preset conan-release -j$(nproc)
    
    - name: Test
      run: ctest --preset conan-release --output-on-failure
```

---

## 获取帮助

如果遇到构建问题，请：

1. 查看 CMake 和 Conan 的输出信息
2. 确认系统要求是否满足
3. 查看 [Conan 文档](https://docs.conan.io/)
4. 查看 [GitHub Issues](https://github.com/your-repo/Mini-Infer/issues)
5. 提交新的 Issue 并附上完整的错误日志

---

## 相关文档

- **[快速开始](../QUICK_START.md)** - 快速上手指南
- **[Conan 构建指南](CONAN_BUILD_GUIDE.md)** - Conan 详细使用说明
- **[CUDA 配置指南](CUDA_CONAN_SETUP.md)** - CUDA 后端配置
- **[入门教程](GETTING_STARTED.md)** - 完整的入门教程
