# Mini-Infer 构建指南

## 系统要求

### Windows
- Visual Studio 2017 或更高版本（推荐 VS 2022）
- CMake 3.18+
- PowerShell

### Linux
- GCC 7+ 或 Clang 5+
- CMake 3.18+
- Make 或 Ninja

### macOS
- Xcode 10+
- CMake 3.18+
- Make 或 Ninja

## 快速构建

### Windows

使用提供的 PowerShell 脚本：

```powershell
# Release 构建
.\build.ps1

# Debug 构建
.\build.ps1 -BuildType Debug

# 清理并构建
.\build.ps1 -Clean

# 构建并运行测试
.\build.ps1 -Test

# 完整构建流程
.\build.ps1 -Clean -Test
```

手动构建：

```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Linux/macOS

使用提供的 Shell 脚本：

```bash
# 添加执行权限（首次）
chmod +x build.sh

# Release 构建
./build.sh

# Debug 构建
./build.sh --debug

# 清理并构建
./build.sh --clean

# 构建并运行测试
./build.sh --test

# 指定并行任务数
./build.sh --jobs 8

# 完整构建流程
./build.sh --clean --test --release
```

手动构建：

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## CMake 配置选项

### 基础选项

```bash
# 指定构建类型
-DCMAKE_BUILD_TYPE=Release|Debug

# 指定安装路径
-DCMAKE_INSTALL_PREFIX=/path/to/install
```

### Mini-Infer 特定选项

```bash
# 是否构建测试（默认 ON）
-DMINI_INFER_BUILD_TESTS=ON|OFF

# 是否构建示例（默认 ON）
-DMINI_INFER_BUILD_EXAMPLES=ON|OFF

# 是否构建共享库（默认 ON）
-DMINI_INFER_BUILD_SHARED_LIBS=ON|OFF

# 是否启用 CUDA 支持（默认 OFF，未来支持）
-DMINI_INFER_ENABLE_CUDA=ON|OFF

# 是否启用性能分析（默认 ON）
-DMINI_INFER_ENABLE_PROFILING=ON|OFF

# 是否启用日志（默认 ON）
-DMINI_INFER_ENABLE_LOGGING=ON|OFF
```

### 示例配置

完整的开发配置：

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMINI_INFER_BUILD_TESTS=ON \
  -DMINI_INFER_BUILD_EXAMPLES=ON \
  -DMINI_INFER_ENABLE_PROFILING=ON \
  -DMINI_INFER_ENABLE_LOGGING=ON
```

生产环境配置：

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMINI_INFER_BUILD_TESTS=OFF \
  -DMINI_INFER_BUILD_EXAMPLES=OFF \
  -DMINI_INFER_ENABLE_LOGGING=OFF
```

## 使用 CMake Presets

项目提供了 `CMakePresets.json` 文件，支持预定义配置：

### Windows

```powershell
# 配置
cmake --preset windows-release

# 构建
cmake --build --preset windows-release

# 测试
ctest --preset windows-release
```

### Linux

```bash
# 配置
cmake --preset linux-release

# 构建
cmake --build --preset linux-release

# 测试
ctest --preset linux-release
```

## 运行测试

构建完成后运行测试：

```bash
cd build

# 运行所有测试
ctest

# 详细输出
ctest --output-on-failure

# 运行特定测试
ctest -R test_tensor

# 并行运行测试
ctest -j4
```

或直接运行测试可执行文件：

```bash
# Windows
.\bin\Release\test_tensor.exe
.\bin\Release\test_backend.exe
.\bin\Release\test_graph.exe

# Linux/macOS
./bin/test_tensor
./bin/test_backend
./bin/test_graph
```

## 运行示例

```bash
# Windows
.\bin\Release\simple_inference.exe
.\bin\Release\build_graph.exe

# Linux/macOS
./bin/simple_inference
./bin/build_graph
```

## 安装

```bash
cd build

# 安装到默认位置
cmake --install .

# 安装到指定位置
cmake --install . --prefix /path/to/install

# Windows 需要指定配置
cmake --install . --config Release
```

安装后的目录结构：

```
install/
├── include/
│   └── mini_infer/
│       ├── core/
│       ├── backends/
│       ├── operators/
│       ├── graph/
│       ├── runtime/
│       └── utils/
├── lib/
│   ├── libmini_infer_core.a
│   ├── libmini_infer_backends.a
│   └── ...
└── lib/cmake/Mini-Infer/
    └── Mini-InferTargets.cmake
```

## 在自己的项目中使用

### 使用 CMake

```cmake
# 方式 1: find_package
find_package(Mini-Infer REQUIRED)
target_link_libraries(your_target PRIVATE MiniInfer::mini_infer_runtime)

# 方式 2: add_subdirectory
add_subdirectory(path/to/Mini-Infer)
target_link_libraries(your_target PRIVATE mini_infer_runtime)
```

### 手动链接

```bash
g++ your_app.cpp \
  -I/path/to/Mini-Infer/include \
  -L/path/to/Mini-Infer/build/lib \
  -lmini_infer_runtime \
  -lmini_infer_graph \
  -lmini_infer_operators \
  -lmini_infer_backends \
  -lmini_infer_core \
  -lmini_infer_utils \
  -lpthread
```

## 常见问题

### 问题 1: CMake 版本太低

```
CMake 3.18 or higher is required
```

**解决方案**: 升级 CMake

```bash
# Ubuntu
sudo apt-get install cmake

# 或从官网下载最新版本
# https://cmake.org/download/
```

### 问题 2: 找不到编译器

**Windows**: 确保安装了 Visual Studio 并添加到 PATH

**Linux**: 安装 GCC 或 Clang

```bash
sudo apt-get install build-essential  # Ubuntu
sudo yum groupinstall "Development Tools"  # CentOS
```

### 问题 3: 链接错误

确保按正确顺序链接库：

```
mini_infer_runtime
mini_infer_graph
mini_infer_operators
mini_infer_backends
mini_infer_core
mini_infer_utils
```

### 问题 4: 头文件找不到

确保包含正确的头文件路径：

```cpp
#include "mini_infer/mini_infer.h"  // 正确
// 而不是
#include "mini_infer.h"  // 错误
```

## 性能优化

### Release 构建优化

确保使用 Release 模式：

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

### GCC/Clang 额外优化

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native"
```

### MSVC 额外优化

```bash
cmake .. \
  -DCMAKE_CXX_FLAGS="/O2 /Ob2 /Oi /Ot /GL"
```

## 清理

```bash
# 清理构建目录
rm -rf build/

# 或使用构建脚本
./build.sh --clean       # Linux/macOS
.\build.ps1 -Clean       # Windows
```

## 持续集成

项目可以轻松集成到 CI/CD 流程：

### GitHub Actions 示例

```yaml
- name: Configure
  run: cmake -B build -DCMAKE_BUILD_TYPE=Release

- name: Build
  run: cmake --build build --parallel

- name: Test
  run: cd build && ctest --output-on-failure
```

## 获取帮助

如果遇到构建问题，请：

1. 查看 CMake 输出的错误信息
2. 确认系统要求是否满足
3. 查看 [GitHub Issues](https://github.com/your-repo/Mini-Infer/issues)
4. 提交新的 Issue 并附上完整的错误日志

