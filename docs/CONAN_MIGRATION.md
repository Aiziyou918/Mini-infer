# 从 vcpkg 迁移到 Conan

本文档记录了 Mini-Infer 从 vcpkg 迁移到 Conan 的完整过程和变更。

## 迁移动机

### vcpkg 的局限性

虽然 vcpkg 是一个优秀的包管理器，但在跨平台项目中存在一些问题：

1. **平台特定的工具链文件路径**
   - Windows: `C:/vcpkg/scripts/buildsystems/vcpkg.cmake`
   - Linux: `/usr/local/vcpkg/...`
   - 需要每个开发者手动配置路径

2. **依赖安装不统一**
   - Windows: `vcpkg install protobuf:x64-windows`
   - Linux: `apt-get install libprotobuf-dev`
   - macOS: `brew install protobuf`

3. **构建命令不一致**
   - 需要针对不同平台编写不同的构建脚本

4. **Abseil 依赖问题**
   - MinGW 环境下需要手动链接 Abseil 组件
   - MSVC 和 GCC 行为不一致

### Conan 的优势

1. **✅ 真正的跨平台**
   ```bash
   # 所有平台使用相同命令
   conan install . --build=missing
   cmake --preset xxx
   cmake --build build/xxx
   ```

2. **✅ 自动依赖管理**
   - 自动下载二进制包或从源码编译
   - 自动处理传递依赖（Protobuf → Abseil）
   - 自动生成 CMake 工具链文件

3. **✅ 可重现构建**
   - 锁定依赖版本（conan.lock）
   - 保证团队构建一致性

4. **✅ 与 CMake 无缝集成**
   - 自动生成 `conan_toolchain.cmake`
   - 自动提供 CMake targets（`protobuf::libprotobuf`）

## 迁移变更列表

### 1. 新增文件

#### Conan 配置文件
- ✅ `conanfile.py` - Conan 包配方
  - 定义项目依赖（Protobuf 3.21.12）
  - 定义构建选项（enable_onnx, enable_logging, enable_cuda）
  - CMake 构建集成

#### 文档和脚本
- ✅ `docs/CONAN_BUILD_GUIDE.md` - Conan 详细使用文档
- ✅ `docs/CONAN_MIGRATION.md` - 迁移记录（本文档）
- ✅ `build_with_conan.ps1` - Windows 自动构建脚本
- ✅ `build_with_conan.sh` - Linux/macOS 自动构建脚本
- ✅ `.conanignore` - Conan 打包时忽略的文件

### 2. 修改文件

#### CMakeLists.txt
**变更前 (vcpkg):**
```cmake
# Find Abseil (required by Protobuf)
find_package(absl CONFIG REQUIRED)

# Find Protobuf using CONFIG mode
find_package(Protobuf REQUIRED CONFIG)
```

**变更后 (Conan):**
```cmake
# Find Protobuf (managed by Conan or system)
find_package(Protobuf REQUIRED CONFIG)
# Abseil 由 Conan 自动处理，无需手动查找
```

#### CMakePresets.json
**变更前 (vcpkg):**
```json
{
  "name": "windows-vcpkg-base",
  "toolchainFile": "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
}
```

**变更后 (Conan):**
```json
{
  "name": "windows-conan-base",
  "toolchainFile": "${sourceDir}/build/${presetName}/generators/conan_toolchain.cmake"
}
```

**预设重命名:**
- `windows-vcpkg-debug` → `windows-conan-debug`
- `windows-vcpkg-release` → `windows-conan-release`
- `linux-onnx-debug` → `linux-conan-debug`
- `linux-onnx-release` → `linux-conan-release`

#### src/importers/CMakeLists.txt
**变更前 (vcpkg + MinGW workaround):**
```cmake
target_link_libraries(mini_infer_importers
    PUBLIC
        protobuf::libprotobuf
        # 显式链接 Abseil 组件（MinGW 需要）
        absl::log_internal_check_op
        absl::log_internal_message
        absl::log_internal_globals
        absl::base
        absl::strings
)
```

**变更后 (Conan):**
```cmake
target_link_libraries(mini_infer_importers
    PUBLIC
        protobuf::libprotobuf
        # Abseil 自动处理，无需显式链接
)
```

#### README.md
- 移除 vcpkg 安装说明
- 添加 Conan 安装和使用说明
- 更新快速开始部分
- 更新 CMake 预设列表

#### .gitignore
- 已有 `conan/` 忽略项（无需修改）

### 3. 删除文件

- ❌ `docs/ONNX_IMPORT_GUIDE.md` - 内容已整合到其他文档

## 使用对比

### vcpkg 方式（旧）

```powershell
# Windows
vcpkg install protobuf:x64-windows
cmake -B build -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release

# Linux (需要不同的命令)
sudo apt-get install libprotobuf-dev
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Conan 方式（新）

```bash
# 所有平台使用相同命令！
conan install . --output-folder=build/xxx --build=missing -s build_type=Release
cmake --preset xxx
cmake --build build/xxx

# 或使用自动脚本
./build_with_conan.sh --type Release    # Linux/macOS
.\build_with_conan.ps1 -BuildType Release  # Windows
```

## 迁移步骤（供其他项目参考）

### 1. 创建 conanfile.py

```python
from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout

class YourProjectConan(ConanFile):
    name = "your-project"
    version = "0.1.0"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"
    
    def requirements(self):
        self.requires("protobuf/3.21.12")
    
    def layout(self):
        cmake_layout(self)
    
    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
```

### 2. 更新 CMakeLists.txt

```cmake
# 移除 vcpkg 特定的查找逻辑
# 使用标准的 find_package
find_package(Protobuf REQUIRED CONFIG)

# 使用现代 CMake targets
target_link_libraries(your_target PRIVATE protobuf::libprotobuf)
```

### 3. 更新 CMakePresets.json

```json
{
  "name": "your-preset",
  "toolchainFile": "${sourceDir}/build/${presetName}/generators/conan_toolchain.cmake"
}
```

### 4. 测试构建

```bash
# 安装依赖
conan install . --output-folder=build/test --build=missing

# 配置和构建
cmake --preset your-preset
cmake --build build/test
```

## 常见问题

### Q: 如何指定 Protobuf 版本？

在 `conanfile.py` 中修改：
```python
def requirements(self):
    self.requires("protobuf/3.21.12")  # 指定版本
```

### Q: 如何添加其他依赖？

```python
def requirements(self):
    self.requires("protobuf/3.21.12")
    self.requires("boost/1.81.0")
    self.requires("opencv/4.5.5")
```

### Q: 如何使用本地缓存加速？

```bash
# Conan 会自动缓存二进制包
# 首次编译慢，后续很快

# 查看缓存
conan cache path protobuf/3.21.12
```

### Q: 团队如何共享配置？

```bash
# 生成 lockfile
conan lock create . --lockfile=conan.lock

# 提交 conan.lock 到版本控制
git add conan.lock

# 其他人使用 lockfile
conan install . --lockfile=conan.lock --build=missing
```

## 性能对比

| 指标 | vcpkg | Conan |
|------|-------|-------|
| 首次配置时间 | ~2-5 分钟 | ~2-5 分钟 |
| 后续配置时间 | ~30 秒 | ~5 秒（缓存） |
| 跨平台一致性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 学习曲线 | 简单 | 中等 |
| 社区支持 | 良好 | 优秀 |

## 总结

### ✅ 成功完成

1. **完全移除 vcpkg 依赖**
   - 无需手动安装 Protobuf
   - 无需配置工具链路径

2. **实现真正的跨平台构建**
   - Windows/Linux/macOS 使用相同命令
   - 一套脚本全平台运行

3. **简化依赖管理**
   - Abseil 等传递依赖自动处理
   - 版本锁定和可重现构建

4. **保持向后兼容**
   - 保留基础构建方式（不带 ONNX）
   - 保留原有的目录结构

### 🎯 未来改进

1. **添加更多依赖**
   - CUDA 支持时添加 CUDA 相关包
   - 添加性能分析工具依赖

2. **持续集成**
   - 在 CI/CD 中使用 Conan
   - 构建缓存优化

3. **Conan 包发布**
   - 将 Mini-Infer 发布到 ConanCenter
   - 方便其他项目引用

## 参考资料

- [Conan 官方文档](https://docs.conan.io/)
- [Conan CMake 集成](https://docs.conan.io/2/reference/tools/cmake.html)
- [从 vcpkg 迁移到 Conan](https://docs.conan.io/2/examples/cross_platform.html)
- [Mini-Infer Conan 构建指南](CONAN_BUILD_GUIDE.md)
