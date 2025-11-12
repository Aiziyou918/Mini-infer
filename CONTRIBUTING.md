# 贡献指南

感谢你对 Mini-Infer 项目的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告问题

如果你发现了 bug 或有功能建议：

1. 先搜索现有的 Issues，确保问题未被报告
2. 创建新 Issue，提供以下信息：
   - 清晰的标题
   - 详细的描述
   - 重现步骤（对于 bug）
   - 预期行为和实际行为
   - 系统信息（操作系统、编译器版本等）
   - 相关日志和错误信息

### 提交代码

1. **Fork 仓库**

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b bugfix/your-bug-fix
   ```

3. **编写代码**
   - 遵循项目的代码风格
   - 添加必要的注释
   - 更新相关文档
   - 添加测试用例

4. **测试**
   ```bash
   ./build.sh --clean --test
   ```

5. **提交**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **推送并创建 Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 代码风格

### C++ 风格指南

我们使用 Google C++ Style Guide 的变体。项目包含 `.clang-format` 文件。

**基本规则**：

```cpp
// 1. 命名规范
class MyClass {};              // PascalCase for classes
void my_function() {}          // snake_case for functions
int my_variable = 0;           // snake_case for variables
const int MAX_SIZE = 100;      // UPPER_CASE for constants

// 2. 缩进：4 个空格
void example() {
    if (condition) {
        do_something();
    }
}

// 3. 花括号：紧跟在语句后
if (condition) {
    // code
} else {
    // code
}

// 4. 指针和引用：符号靠左
int* ptr;
int& ref;

// 5. 注释：使用 Doxygen 风格
/**
 * @brief Brief description
 * @param param1 Description
 * @return Return value description
 */
```

### 格式化代码

使用 clang-format 自动格式化：

```bash
# 格式化单个文件
clang-format -i src/core/tensor.cpp

# 格式化所有文件
find src include -name "*.cpp" -o -name "*.h" | xargs clang-format -i
```

## 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**类型**：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 添加测试
- `chore`: 构建或工具变动

**示例**：

```
feat(core): add FP16 support for Tensor

- Add FLOAT16 data type
- Implement conversion functions
- Update allocator to support FP16

Closes #123
```

## 开发工作流

### 1. 环境设置

```bash
# 克隆仓库
git clone https://github.com/your-repo/Mini-Infer.git
cd Mini-Infer

# 构建项目
./build.sh --debug

# 运行测试
./build.sh --test
```

### 2. 开发流程

```bash
# 创建功能分支
git checkout -b feature/my-feature

# 编写代码
# ... 编辑文件 ...

# 构建并测试
./build.sh --clean --test

# 提交更改
git add .
git commit -m "feat: add my feature"

# 推送
git push origin feature/my-feature
```

### 3. 代码审查

Pull Request 会经过以下检查：

- [ ] 代码符合风格指南
- [ ] 所有测试通过
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 没有引入新的警告

## 添加新功能

### 添加新算子

1. **创建头文件** `include/mini_infer/operators/my_op.h`

```cpp
#pragma once
#include "mini_infer/operators/operator.h"

namespace mini_infer {
namespace operators {

class MyOperator : public Operator {
public:
    MyOperator();
    
    core::Status forward(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs
    ) override;
    
    core::Status infer_shape(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes
    ) override;
};

} // namespace operators
} // namespace mini_infer
```

2. **创建实现文件** `src/operators/my_op.cpp`

3. **添加到 CMakeLists.txt**

```cmake
set(OPERATORS_SOURCES
    operator.cpp
    conv2d.cpp
    my_op.cpp  # 添加这行
)
```

4. **添加测试** `tests/test_my_op.cpp`

5. **更新文档** `docs/API.md`

### 添加新后端

1. 实现 `Backend` 接口
2. 在 `BackendFactory` 中注册
3. 添加相应的测试
4. 更新文档

## 测试

### 编写测试

测试文件放在 `tests/` 目录：

```cpp
#include "mini_infer/core/tensor.h"
#include <cassert>

void test_my_feature() {
    // Arrange
    core::Shape shape({2, 3});
    auto tensor = core::Tensor::create(shape, core::DataType::FLOAT32);
    
    // Act
    // ... 执行操作 ...
    
    // Assert
    assert(condition);
}

int main() {
    try {
        test_my_feature();
        std::cout << "✓ Test passed" << std::endl;
        return 0;
    } catch (...) {
        std::cerr << "✗ Test failed" << std::endl;
        return 1;
    }
}
```

### 运行测试

```bash
# 运行所有测试
./build.sh --test

# 运行特定测试
cd build
./bin/test_tensor
```

## 文档

### 代码文档

使用 Doxygen 风格注释：

```cpp
/**
 * @brief Create a new tensor
 * 
 * @param shape The shape of the tensor
 * @param dtype The data type of the tensor
 * @return std::shared_ptr<Tensor> Pointer to the created tensor
 * 
 * @note This function allocates memory for the tensor
 * @see Tensor::reshape()
 */
static std::shared_ptr<Tensor> create(const Shape& shape, DataType dtype);
```

### Markdown 文档

- `README.md`: 项目概览
- `docs/API.md`: API 参考
- `docs/ARCHITECTURE.md`: 架构设计
- `docs/BUILD.md`: 构建指南

## Pull Request 检查清单

提交 PR 前请确认：

- [ ] 代码已格式化 (`clang-format`)
- [ ] 所有测试通过
- [ ] 添加了新功能的测试
- [ ] 更新了相关文档
- [ ] 提交信息符合规范
- [ ] 没有不必要的文件（构建产物等）
- [ ] PR 描述清晰，说明了改动内容

## 社区准则

- 尊重他人
- 欢迎新手
- 建设性的讨论
- 专注于技术问题

## 获取帮助

- 查看现有文档和示例
- 搜索已关闭的 Issues
- 在 Discussions 中提问
- 联系维护者

## 许可证

贡献的代码将使用 MIT 许可证发布。

---

感谢你的贡献！🎉

