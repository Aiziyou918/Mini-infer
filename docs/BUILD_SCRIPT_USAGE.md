# 构建脚本使用指南

## 📋 概述

`build.ps1` (Windows) 和 `build.sh` (Linux/macOS) 是 Mini-Infer 的自动化构建脚本，提供智能化的一键式构建体验。

## 🌟 核心特性

### 1. 智能 Ninja 检测

脚本会自动检测系统中是否安装了 Ninja 构建工具：

- ✅ **已安装 Ninja**: 自动使用，编译速度提升 50%+
- ❌ **未安装 Ninja**: 询问用户是否安装
  - 选择 `Y`: 显示安装命令，等待用户安装后继续
  - 选择 `n`: 回退到默认生成器（Visual Studio 或 Unix Makefiles）

### 2. 一键式构建

无需记忆复杂的命令，一个脚本完成所有步骤：
1. 检查 Conan 安装
2. 检测 Ninja 生成器
3. 安装依赖（调用 `conan install`）
4. 配置 CMake
5. 编译项目
6. 可选：运行测试
7. 可选：安装到 install 目录

### 3. 完整的错误处理

- 每个步骤都有错误检测
- 失败时提供清晰的错误信息
- 彩色输出便于识别状态

## 🪟 Windows (build.ps1)

### 基本用法

```powershell
# 最简单的方式 - Debug 构建 + ONNX
.\build.ps1

# Release 构建
.\build.ps1 -BuildType Release

# 清理后重新构建
.\build.ps1 -Clean

# 构建并运行测试
.\build.ps1 -Test

# 完整流程：清理 + Release + 测试 + 安装
.\build.ps1 -BuildType Release -Clean -Test -Install
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-BuildType` | String | `Debug` | 构建类型（`Debug` 或 `Release`） |
| `-EnableOnnx` | Switch | `$true` | 启用 ONNX 支持 |
| `-EnableLogging` | Switch | `$true` | 启用日志输出 |
| `-EnableCuda` | Switch | `$false` | 启用 CUDA（未来支持） |
| `-Clean` | Switch | `$false` | 清理构建目录 |
| `-Test` | Switch | `$false` | 运行测试 |
| `-Install` | Switch | `$false` | 安装到 install 目录 |

### 实际运行示例

```powershell
PS I:\code\Mini-Infer> .\build.ps1
[INFO] =========================================
[INFO] Mini-Infer 自动化构建脚本 (Conan)
[INFO] =========================================
[INFO] 构建类型: Debug
[INFO] ONNX支持: True
[INFO] 日志支持: True
[INFO] CUDA支持: False

[INFO] 检查 Conan 安装...
[SUCCESS] 找到 Conan: Conan version 2.0.13

[INFO] 检查 Ninja 生成器...

未检测到 Ninja 生成器

Ninja 可以显著提升编译速度（提升 50%+）
是否要安装 Ninja？[Y/n]: Y

[INFO] 尝试安装 Ninja...

请选择安装方式:
  1. scoop install ninja     (推荐)
  2. choco install ninja
  3. pip install ninja

请在另一个终端执行安装命令，完成后按回车继续...
(用户在另一个终端运行: scoop install ninja)
<按回车>

[SUCCESS] Ninja 安装成功: 1.11.1
[INFO] 安装依赖...
[INFO] 使用 Ninja 生成器
... (Conan 安装输出)
[SUCCESS] 依赖安装完成

[INFO] 配置 CMake...
... (CMake 配置输出)
[SUCCESS] CMake 配置完成

[INFO] 编译项目...
... (编译输出)
[SUCCESS] 编译完成

[SUCCESS] =========================================
[SUCCESS] 构建成功完成!
[SUCCESS] =========================================

[INFO] 二进制文件: build\Debug\bin\

[INFO] 运行示例:
  .\build\Debug\bin\onnx_parser_example.exe .\models\python\lenet5\models\lenet5.onnx
```

## 🐧 Linux/macOS (build.sh)

### 基本用法

```bash
# 最简单的方式 - Debug 构建 + ONNX
./build.sh

# Release 构建
./build.sh -r

# 清理后重新构建
./build.sh -c

# 构建并运行测试
./build.sh -t

# 完整流程：清理 + Release + 测试 + 安装
./build.sh -r -c -t -i

# 禁用 ONNX
./build.sh --no-onnx

# 启用 CUDA（未来支持）
./build.sh --enable-cuda
```

### 参数说明

| 参数 | 别名 | 默认值 | 说明 |
|------|------|--------|------|
| `-d` | `--debug` | 是 | Debug 构建 |
| `-r` | `--release` | - | Release 构建 |
| `-c` | `--clean` | - | 清理构建目录 |
| `-t` | `--test` | - | 运行测试 |
| `-i` | `--install` | - | 安装 |
| - | `--no-onnx` | - | 禁用 ONNX |
| - | `--no-logging` | - | 禁用日志 |
| - | `--enable-cuda` | - | 启用 CUDA |
| `-h` | `--help` | - | 显示帮助 |

### 实际运行示例

```bash
$ ./build.sh -r
[INFO] =========================================
[INFO] Mini-Infer 自动化构建脚本 (Conan)
[INFO] =========================================
[INFO] 构建类型: Release
[INFO] ONNX支持: true
[INFO] 日志支持: true
[INFO] CUDA支持: false

[INFO] 检查 Conan 安装...
[SUCCESS] 找到 Conan: Conan version 2.0.13

[INFO] 检查 Ninja 生成器...

未检测到 Ninja 生成器

Ninja 可以显著提升编译速度（提升 50%+）
是否要安装 Ninja？[Y/n]: n

[INFO] 跳过 Ninja 安装，使用默认生成器 (Unix Makefiles)
[INFO] 安装依赖...
[INFO] 使用 Unix Makefiles 生成器
... (Conan 安装输出)
[SUCCESS] 依赖安装完成

[INFO] 配置 CMake...
... (CMake 配置输出)
[SUCCESS] CMake 配置完成

[INFO] 编译项目...
... (编译输出)
[SUCCESS] 编译完成

[SUCCESS] =========================================
[SUCCESS] 构建成功完成!
[SUCCESS] =========================================

[INFO] 二进制文件: build/Release/bin/

[INFO] 运行示例:
  ./build/Release/bin/onnx_parser_example ./models/python/lenet5/models/lenet5.onnx
```

## 🎯 使用场景

### 场景 1: 日常开发（Debug）

```bash
# 快速迭代，启用所有调试功能
./build.sh             # Linux/macOS
.\build.ps1            # Windows
```

### 场景 2: 性能测试（Release）

```bash
# 优化构建，测试性能
./build.sh -r          # Linux/macOS
.\build.ps1 -BuildType Release  # Windows
```

### 场景 3: 持续集成（CI）

```bash
# 清理 + Release + 测试
./build.sh -r -c -t    # Linux/macOS
.\build.ps1 -BuildType Release -Clean -Test  # Windows
```

### 场景 4: 最小化构建

```bash
# 不需要 ONNX，加快编译
./build.sh --no-onnx --no-logging
```

## 💡 技巧和窍门

### Ninja 安装建议

**Windows (推荐 scoop):**
```powershell
# 如果没有 scoop，先安装 scoop
iex "& {$(irm get.scoop.sh)} -RunAsAdmin"

# 安装 Ninja
scoop install ninja
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ninja-build
```

**macOS:**
```bash
brew install ninja
```

**通用方法 (pip):**
```bash
pip install ninja
```

### 跳过交互式提示

如果你不想被询问是否安装 Ninja，可以：

1. **提前安装 Ninja**（推荐）
2. **直接选择 `n`**，使用默认生成器

### 查看详细输出

脚本会显示所有步骤的彩色输出：
- 🔵 `[INFO]` - 信息
- ✅ `[SUCCESS]` - 成功
- ❌ `[ERROR]` - 错误

## ❓ 常见问题

### Q: 脚本卡在 "请在另一个终端执行安装命令" 怎么办？

**A**: 打开一个新的终端窗口，执行提示的安装命令（如 `scoop install ninja`），安装完成后回到原终端按回车。

### Q: 不想安装 Ninja，直接跳过可以吗？

**A**: 可以！当询问时输入 `n`，脚本会使用默认生成器继续构建。

### Q: 已经安装了 Ninja，为什么还是检测不到？

**A**: 确保 `ninja` 在 PATH 环境变量中。运行 `ninja --version` 测试。

### Q: 如何禁用所有可选功能？

**A**: 
```bash
# Linux/macOS
./build.sh --no-onnx --no-logging

# Windows (目前还需要修改 conanfile.py 的默认值)
# 或手动构建时使用 -o enable_onnx=False
```

## 📚 相关文档

- [快速开始指南](../QUICK_START.md)
- [Conan 构建指南](CONAN_BUILD_GUIDE.md)
- [Conan 选项指南](CONAN_OPTIONS_GUIDE.md)
- [README](../README.md)
