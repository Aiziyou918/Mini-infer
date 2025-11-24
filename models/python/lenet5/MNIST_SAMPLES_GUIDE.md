# MNIST 测试样本导出指南

本指南介绍如何导出 MNIST 测试集样本，用于在 Mini-Infer C++ 推理引擎中测试和验证模型。

## 快速开始

### 1. 基本导出

```bash
# 导出 10 个样本（每个类别 1 个）
python export_mnist_samples.py

# 输出到 ./test_samples/ 目录
```

### 2. 自定义数量

```bash
# 导出 100 个样本
python export_mnist_samples.py --num-samples 100

# 每个类别 10 个样本（总共 100 个）
python export_mnist_samples.py --num-per-class 10
```

### 3. 选择特定类别

```bash
# 只导出数字 0, 1, 2
python export_mnist_samples.py --classes 0 1 2 --num-per-class 5

# 导出数字 7, 8, 9
python export_mnist_samples.py --classes 7 8 9 --num-samples 30
```

---

## 导出格式

### Binary Format (.bin)

**用途**: C++ 直接加载，性能最优

**格式**:
- 数据类型: `float32`
- 形状: `[1, 28, 28]` (C, H, W)
- 归一化: `(pixel - 0.1307) / 0.3081`
- 字节序: Little-endian

**大小**: 每个样本 3.1 KB (1×28×28×4 bytes)

### PNG Format (.png)

**用途**: 可视化检查，调试

**格式**:
- 格式: 标准 PNG 图像
- 尺寸: 28×28 像素
- 颜色: 灰度
- 原始值: 0-255 (未归一化)

**文件名**: `sample_XXXX_label_Y.png`
- XXXX: 样本索引 (0000-9999)
- Y: 真实标签 (0-9)

### NumPy Format (.npy)

**用途**: Python 调试和验证

**格式**:
- NumPy 标准格式
- 已归一化的 float32 数组
- 可用 `np.load()` 加载

---

## 使用示例

### 示例 1: 基础测试集

```bash
# 导出 10 个样本（每类 1 个）
python export_mnist_samples.py \
    --num-per-class 1 \
    --output-dir ./test_samples_basic \
    --formats binary png
```

**输出**:
```
test_samples_basic/
├── binary/
│   ├── sample_0000.bin
│   ├── sample_0001.bin
│   └── ...
├── images/
│   ├── sample_0000_label_3.png
│   ├── sample_0001_label_7.png
│   └── ...
├── samples_metadata.json
└── mnist_loader.h
```

### 示例 2: 完整测试集

```bash
# 导出 1000 个样本（每类 100 个）
python export_mnist_samples.py \
    --num-per-class 100 \
    --output-dir ./test_samples_full \
    --seed 2024
```

### 示例 3: 困难样本

```bash
# 手动筛选后导出特定索引
# （可以先运行模型找出预测错误的样本）
python export_mnist_samples.py \
    --classes 3 5 8 \
    --num-per-class 20 \
    --output-dir ./test_samples_hard
```

---

## C++ 集成

### 1. 包含生成的头文件

```cpp
#include "test_samples/mnist_loader.h"

using namespace mnist_loader;
```

### 2. 加载单个样本

```cpp
// 加载一个样本
auto input = load_sample("test_samples/binary/sample_0000.bin");

// input 是 shared_ptr<Tensor>, shape [1, 1, 28, 28]
std::cout << "Input shape: " << input->shape() << std::endl;
```

### 3. 批量测试

```cpp
#include "test_samples/mnist_loader.h"
#include "lenet5_model.h"  // 你的模型

int main() {
    // 加载模型
    auto model = load_lenet5("weights/");
    
    // 获取所有测试样本
    auto samples = mnist_loader::get_test_samples();
    
    int correct = 0;
    int total = samples.size();
    
    for (const auto& sample_info : samples) {
        // 加载输入
        auto input = mnist_loader::load_sample(sample_info.binary_path);
        
        // 推理
        auto output = model->forward({input});
        
        // 获取预测
        int predicted = argmax(output[0]);
        int ground_truth = sample_info.label;
        
        if (predicted == ground_truth) {
            correct++;
            std::cout << "✓ ";
        } else {
            std::cout << "✗ ";
        }
        
        std::cout << "Sample " << sample_info.index 
                  << ": pred=" << predicted 
                  << ", gt=" << ground_truth << std::endl;
    }
    
    float accuracy = 100.0f * correct / total;
    std::cout << "\nAccuracy: " << correct << "/" << total 
              << " (" << accuracy << "%)" << std::endl;
    
    return 0;
}
```

### 4. 完整示例程序

参见 `examples/lenet5_inference.cpp`

---

## 元数据文件

### samples_metadata.json

```json
{
  "total_samples": 10,
  "formats": ["binary", "png"],
  "shape": [1, 28, 28],
  "dtype": "float32",
  "normalization": {
    "mean": 0.1307,
    "std": 0.3081
  },
  "class_distribution": {
    "0": 1,
    "1": 1,
    ...
  },
  "samples": [
    {
      "index": 0,
      "label": 7,
      "shape": [1, 28, 28],
      "files": {
        "binary": "binary/sample_0000.bin",
        "png": "images/sample_0000_label_7.png"
      },
      "binary_info": {
        "dtype": "float32",
        "shape": [1, 28, 28],
        "size_bytes": 3136,
        "mean": -0.0042,
        "std": 0.9876,
        "min": -0.4242,
        "max": 2.8215
      }
    },
    ...
  ]
}
```

---

## 高级用法

### 1. 导出所有格式

```bash
python export_mnist_samples.py \
    --num-samples 50 \
    --formats binary png npy \
    --output-dir ./test_samples_all_formats
```

### 2. 只导出二进制（最快）

```bash
python export_mnist_samples.py \
    --num-per-class 100 \
    --formats binary \
    --output-dir ./test_samples_binary_only
```

### 3. 使用不同随机种子

```bash
# 生成不同的测试集
python export_mnist_samples.py --seed 42 --output-dir ./test_set_1
python export_mnist_samples.py --seed 123 --output-dir ./test_set_2
python export_mnist_samples.py --seed 456 --output-dir ./test_set_3
```

---

## Python 验证

### 加载并验证导出的样本

```python
import numpy as np
from PIL import Image

# 加载二进制文件
data = np.fromfile('test_samples/binary/sample_0000.bin', dtype=np.float32)
data = data.reshape(1, 28, 28)

print(f"Shape: {data.shape}")
print(f"Mean: {data.mean():.4f}")
print(f"Std: {data.std():.4f}")
print(f"Min: {data.min():.4f}, Max: {data.max():.4f}")

# 加载 PNG（可视化）
img = Image.open('test_samples/images/sample_0000_label_7.png')
img.show()

# 加载 NumPy
npy_data = np.load('test_samples/numpy/sample_0000.npy')
assert np.allclose(data, npy_data), "Data mismatch!"
```

### 与 PyTorch 模型对比

```python
import torch
from lenet5_model import LeNet5

# 加载模型
model = LeNet5()
checkpoint = torch.load('checkpoints/lenet5_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载样本
data = np.fromfile('test_samples/binary/sample_0000.bin', dtype=np.float32)
tensor = torch.from_numpy(data).reshape(1, 1, 28, 28)

# 推理
with torch.no_grad():
    output = model(tensor)
    predicted = output.argmax(dim=1).item()

print(f"Predicted: {predicted}")
```

---

## 性能数据

| 样本数 | Binary 大小 | PNG 大小 | 导出时间 |
|--------|------------|---------|---------|
| 10 | 31 KB | ~50 KB | <1s |
| 100 | 310 KB | ~500 KB | ~2s |
| 1000 | 3.1 MB | ~5 MB | ~15s |
| 10000 | 31 MB | ~50 MB | ~2min |

*测试环境: Intel i7, SSD*

---

## 文件结构

```
test_samples/
├── binary/              # 二进制文件（推理用）
│   ├── sample_0000.bin
│   ├── sample_0001.bin
│   └── ...
├── images/              # PNG 图片（可视化）
│   ├── sample_0000_label_7.png
│   ├── sample_0001_label_2.png
│   └── ...
├── numpy/               # NumPy 文件（调试用）
│   ├── sample_0000.npy
│   ├── sample_0001.npy
│   └── ...
├── samples_metadata.json   # 元数据
└── mnist_loader.h          # C++ 加载器（自动生成）
```

---

## 常见问题

### Q1: 如何选择导出数量？

**A**: 根据测试需求：
- **快速验证**: 10-50 个样本
- **准确率测试**: 100-1000 个样本
- **完整评估**: 10000 个样本（完整测试集）

### Q2: Binary 和 PNG 有什么区别？

**A**: 
- **Binary**: 已归一化，可直接用于推理，加载快
- **PNG**: 原始图像（0-255），用于可视化，需要在 C++ 中归一化

### Q3: 如何添加更多预处理？

**A**: 修改 `export_mnist_samples.py` 中的 `transform`：

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    # 添加你的变换
    transforms.RandomRotation(5),
])
```

### Q4: 可以导出训练集吗？

**A**: 可以，修改脚本中的 `train=False` 改为 `train=True`

### Q5: 如何验证导出正确性？

**A**: 使用 Python 验证：

```bash
# 在脚本中添加验证选项（未来功能）
python export_mnist_samples.py --verify
```

---

## 与 C++ 推理流程

```
1. 训练模型
   └─> python train_lenet5.py
   
2. 导出权重
   └─> python export_lenet5.py --format weights
   
3. 导出测试样本
   └─> python export_mnist_samples.py --num-per-class 10
   
4. C++ 推理
   ├─> 加载权重 (load_weights())
   ├─> 加载样本 (mnist_loader::load_sample())
   ├─> 运行推理 (model->forward())
   └─> 计算准确率
```

---

## 扩展建议

### 1. 导出困难样本

训练后找出模型预测错误的样本并单独导出：

```python
# 伪代码
misclassified_indices = find_misclassified(model, test_set)
export_samples(misclassified_indices, "hard_samples/")
```

### 2. 数据增强测试

导出带旋转、噪声等增强的样本：

```bash
python export_mnist_samples.py \
    --augment rotate,noise \
    --output-dir ./augmented_samples
```

### 3. 批量推理

将多个样本打包成一个文件，支持批量推理：

```bash
python export_mnist_samples.py \
    --batch-size 32 \
    --output-format batched \
    --output-dir ./batched_samples
```

---

## 参考资料

- **MNIST 数据集**: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- **LeNet-5 论文**: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- **Mini-Infer 文档**: `docs/`
- **C++ 推理示例**: `examples/lenet5_inference.cpp`
