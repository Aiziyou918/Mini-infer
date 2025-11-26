# Conan 选项自动传递指南

## 📋 概述

从现在开始，你在 `conan install` 时指定的选项会**自动传递**到生成的 `CMakePresets.json` 中，无需手动配置！

## 🔄 工作流程

### 传统方式（已弃用）

```bash
# 1. 运行 conan install
conan install . --output-folder=build/xxx --build=missing

# 2. 手动编辑 CMakePresets.json
# 需要手动添加：
# "MINI_INFER_ENABLE_ONNX": "ON"
# "MINI_INFER_ENABLE_LOGGING": "ON"
# 等等...

# 3. 配置 CMake
cmake --preset xxx
```

### 🚀 新方式（自动化）

```bash
# 1. 运行 conan install 时指定选项
conan install . \
  --output-folder=build/xxx \
  -o enable_onnx=True \      # ← 自动转换为 MINI_INFER_ENABLE_ONNX=ON
  -o enable_logging=False \  # ← 自动转换为 MINI_INFER_ENABLE_LOGGING=OFF
  --build=missing

# 2. 直接配置 CMake（选项已自动设置）
cmake --preset conan-debug  # 使用 Conan 生成的预设

# 3. 编译
cmake --build build/Debug
```

## 🎯 选项映射表

| Conan 选项 | CMake 变量 | 默认值 | 说明 |
|-----------|-----------|--------|------|
| `-o enable_onnx=True` | `MINI_INFER_ENABLE_ONNX=ON` | `True` | ONNX 模型导入支持 |
| `-o enable_onnx=False` | `MINI_INFER_ENABLE_ONNX=OFF` | | 禁用 ONNX（不安装 Protobuf） |
| `-o enable_logging=True` | `MINI_INFER_ENABLE_LOGGING=ON` | `True` | 日志输出支持 |
| `-o enable_logging=False` | `MINI_INFER_ENABLE_LOGGING=OFF` | | 禁用日志（性能优化） |
| `-o enable_cuda=True` | `MINI_INFER_ENABLE_CUDA=ON` | `False` | CUDA GPU 加速（未来） |
| `-o enable_cuda=False` | `MINI_INFER_ENABLE_CUDA=OFF` | | 仅 CPU 模式 |

## 📂 生成的文件结构

```bash
conan install . --output-folder=build/Debug -o enable_onnx=True --build=missing
```

生成的 `build/Debug/generators/CMakePresets.json`:

```json
{
    "version": 3,
    "configurePresets": [
        {
            "name": "conan-debug",
            "generator": "Ninja",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug",
                "CMAKE_C_COMPILER": "cl",
                "CMAKE_CXX_COMPILER": "cl",
                "MINI_INFER_ENABLE_ONNX": "ON",      // ← 自动添加！
                "MINI_INFER_ENABLE_LOGGING": "ON",   // ← 自动添加！
                "MINI_INFER_ENABLE_CUDA": "OFF"      // ← 自动添加！
            },
            "toolchainFile": "generators/conan_toolchain.cmake"
        }
    ]
}
```

## 💡 常见使用场景

### 场景 1: 完整功能开发（默认）

```bash
# 启用所有功能
conan install . \
  --output-folder=build/dev \
  -s build_type=Debug \
  --build=missing

# 结果：
# - ONNX: ON
# - Logging: ON
# - CUDA: OFF
```

### 场景 2: 最小化构建（CI/测试）

```bash
# 禁用可选功能，加快编译
conan install . \
  --output-folder=build/minimal \
  -s build_type=Release \
  -o enable_onnx=False \
  -o enable_logging=False \
  --build=missing

# 结果：
# - ONNX: OFF (不安装 Protobuf，节省时间)
# - Logging: OFF (减少二进制大小)
# - CUDA: OFF
```

### 场景 3: 生产优化构建

```bash
# 启用 ONNX，禁用日志
conan install . \
  --output-folder=build/production \
  -s build_type=Release \
  -o enable_onnx=True \
  -o enable_logging=False \
  --build=missing

# 结果：
# - ONNX: ON (支持模型加载)
# - Logging: OFF (性能优化)
# - CUDA: OFF
```

### 场景 4: GPU 加速构建（未来）

```bash
# 启用所有功能包括 CUDA
conan install . \
  --output-folder=build/gpu \
  -s build_type=Release \
  -o enable_onnx=True \
  -o enable_cuda=True \
  --build=missing

# 结果：
# - ONNX: ON
# - Logging: ON
# - CUDA: ON
```

## 🔧 技术实现

### conanfile.py 的 generate() 方法

```python
def generate(self):
    from conan.tools.cmake import CMakeToolchain
    
    tc = CMakeToolchain(self)
    
    # 将 Conan 选项转换为 CMake 缓存变量
    tc.cache_variables["MINI_INFER_ENABLE_ONNX"] = "ON" if self.options.enable_onnx else "OFF"
    tc.cache_variables["MINI_INFER_ENABLE_LOGGING"] = "ON" if self.options.enable_logging else "OFF"
    tc.cache_variables["MINI_INFER_ENABLE_CUDA"] = "ON" if self.options.enable_cuda else "OFF"
    
    tc.generate()
```

这个方法在 `conan install` 时自动执行，将选项写入：
1. `conan_toolchain.cmake` - CMake 工具链文件
2. `CMakePresets.json` - CMake 预设文件

## 🎓 最佳实践

### ✅ 推荐做法

```bash
# 1. 使用 Conan 选项控制功能
conan install . -o enable_onnx=True --build=missing

# 2. 使用 Conan 生成的预设
cmake --preset conan-debug

# 3. 不要手动修改生成的 CMakePresets.json
```

### ❌ 不推荐做法

```bash
# 不要忽略 Conan 选项，然后手动修改 CMakePresets.json
conan install . --build=missing
# 然后手动编辑 build/Debug/generators/CMakePresets.json
```

## 📊 选项组合参考

| 用途 | enable_onnx | enable_logging | enable_cuda | 编译时间 | 二进制大小 |
|------|------------|----------------|-------------|----------|-----------|
| **完整开发** | ✅ | ✅ | ❌ | ~2 分钟 | ~5 MB |
| **最小测试** | ❌ | ❌ | ❌ | ~1 分钟 | ~2 MB |
| **生产部署** | ✅ | ❌ | ❌ | ~2 分钟 | ~4 MB |
| **GPU 加速** | ✅ | ✅ | ✅ | ~3 分钟 | ~8 MB |

## 🚀 快速参考

```bash
# 查看当前默认选项
conan inspect . --format=compact

# 列出所有可用选项
conan inspect . --format=json | jq '.options'

# 安装并立即构建
conan install . --output-folder=build/test -o enable_onnx=True --build=missing
cmake --preset conan-debug
cmake --build build/Debug

# 清理并重新配置
rm -rf build/
conan install . --output-folder=build/new -o enable_logging=False --build=missing
```

## 🔗 相关资源

- [Conan 构建指南](CONAN_BUILD_GUIDE.md)
- [Conan 迁移文档](CONAN_MIGRATION.md)
- [ONNX 解析器设计](ONNX_PARSER_DESIGN.md)
