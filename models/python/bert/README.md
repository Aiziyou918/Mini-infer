# Tiny-BERT Model Export and Inference

本目录包含 Tiny-BERT 模型的导出脚本和测试样本，用于 Mini-Infer 推理框架的 BERT 推理示例。

## 模型信息

- **模型**: [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny)
- **参数量**: ~4.4M
- **隐藏层大小**: 128
- **层数**: 2
- **注意力头数**: 2
- **任务**: 文本分类（情感分析，2分类）

## 目录结构

```
bert/
├── README.md                      # 本文件
├── __init__.py
├── export_bert.py                 # 模型导出脚本
├── export_bert_samples.py         # 测试样本导出脚本
├── generate_reference_outputs.py  # 生成 PyTorch 参考输出
├── models/                        # ONNX 模型存放目录
│   └── bert_tiny.onnx
└── test_samples/                  # 测试样本
    ├── binary/                    # 二进制格式输入
    │   ├── sample_0000_input_ids.bin
    │   ├── sample_0000_attention_mask.bin
    │   └── sample_0000_token_type_ids.bin
    ├── samples_metadata.json      # 样本元数据
    └── reference_outputs.json     # PyTorch 参考输出
```

## 环境准备

确保已安装必要的 Python 依赖：

```bash
conda activate mini-infer
pip install transformers torch onnx onnxruntime
```

## 使用方法

### 1. 导出 ONNX 模型

```bash
python export_bert.py --output ./models/bert_tiny.onnx
```

可选参数：
- `--model`: HuggingFace 模型名称（默认: prajjwal1/bert-tiny）
- `--opset`: ONNX opset 版本（默认: 14）
- `--max-seq-length`: 最大序列长度（默认: 128）
- `--num-labels`: 分类标签数量（默认: 2）
- `--verbose`: 启用详细输出

### 2. 导出测试样本

```bash
python export_bert_samples.py --output-dir ./test_samples
```

可选参数：
- `--model`: 用于 tokenizer 的模型名称
- `--max-seq-length`: 最大序列长度（默认: 128）
- `--num-samples`: 导出样本数量（默认: 10）

### 3. 生成参考输出

```bash
python generate_reference_outputs.py --samples-dir ./test_samples
```

这将生成 `test_samples/reference_outputs.json`，包含 PyTorch 推理的参考结果。

### 4. 运行 C++ 推理

```bash
# 编译 Mini-Infer
cmake --build build/Debug --parallel 32

# 运行 BERT 推理示例
./build/Debug/bin/bert_inference \
    models/python/bert/models/bert_tiny.onnx \
    models/python/bert/test_samples \
    10
```

## 输入输出格式

### 输入

| 名称 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| input_ids | [batch, seq_len] | INT64 | Token ID 序列 |
| attention_mask | [batch, seq_len] | INT64 | 注意力掩码 |
| token_type_ids | [batch, seq_len] | INT64 | Token 类型 ID |

### 输出

| 名称 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| logits | [batch, num_labels] | FLOAT32 | 分类 logits |

## 测试样本

测试样本为情感分类任务的文本：

| 标签 | 含义 | 示例 |
|------|------|------|
| 0 | 负面 | "Terrible film, waste of time." |
| 1 | 正面 | "This movie is wonderful!" |

## 验证

对比 Mini-Infer 输出与 `reference_outputs.json` 中的 PyTorch 参考输出：

1. 预测类别应一致
2. Logits 数值差异应 < 1e-4

## 注意事项

1. **数据类型**: BERT 输入为 INT64，确保 C++ 端正确处理
2. **序列长度**: 默认 128，可根据需要调整
3. **模型大小**: bert-tiny 约 17MB（ONNX 格式）
4. **首次运行**: 会自动下载模型和 tokenizer

## 故障排除

### 模型下载失败

如果 HuggingFace 下载失败，可以设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### ONNX 验证失败

如果 ONNX 与 PyTorch 输出差异较大，尝试：
- 降低 opset 版本
- 检查 transformers 版本兼容性
