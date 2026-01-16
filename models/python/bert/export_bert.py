#!/usr/bin/env python3
"""
Export Tiny-BERT Model to ONNX

使用 HuggingFace transformers 的 prajjwal1/bert-tiny 模型
导出为 ONNX 格式，支持文本分类任务

Usage:
    python export_bert.py --output ./models/bert_tiny.onnx
    python export_bert.py --output ./models/bert_tiny.onnx --opset 14 --verbose
"""

import argparse
from pathlib import Path
import torch
import onnx
import numpy as np

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("Warning: onnxruntime not installed, skipping ONNX verification")

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification


def export_bert_to_onnx(
    model_name: str = "prajjwal1/bert-tiny",
    output_path: str = "./models/bert_tiny.onnx",
    opset_version: int = 14,
    max_seq_length: int = 128,
    num_labels: int = 2,
    verbose: bool = False,
    seed: int = 42
) -> str:
    """
    导出 BERT 模型到 ONNX 格式

    Args:
        model_name: HuggingFace 模型名称
        output_path: ONNX 输出路径
        opset_version: ONNX opset 版本
        max_seq_length: 最大序列长度
        num_labels: 分类标签数量
        verbose: 是否输出详细信息
        seed: 随机种子，确保 classifier 权重可复现

    Returns:
        导出的 ONNX 文件路径
    """
    print(f"Loading model: {model_name}")

    # 设置随机种子，确保 classifier 层权重可复现
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 加载模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 文本分类任务
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    model.eval()

    # 保存完整模型权重（包括随机初始化的 classifier）
    output_path = Path(output_path)
    weights_path = output_path.parent / "bert_tiny_weights.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to: {weights_path}")

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Num layers: {model.config.num_hidden_layers}")
    print(f"Num attention heads: {model.config.num_attention_heads}")

    # 创建示例输入
    dummy_input_ids = torch.ones(1, max_seq_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_seq_length, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(1, max_seq_length, dtype=torch.long)

    # 定义输入输出名称
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    output_names = ["logits"]

    # 定义动态轴（支持动态 batch size 和序列长度）
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }

    # 创建输出目录
    output_path_obj = Path(output_path) if isinstance(output_path, str) else output_path
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # 导出 ONNX
    print(f"\nExporting to ONNX (opset {opset_version})...")

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=verbose
    )

    print(f"Model exported to: {output_path}")

    # 获取文件大小
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")

    # 验证 ONNX 模型
    verify_onnx_model(
        str(output_path), model,
        dummy_input_ids, dummy_attention_mask, dummy_token_type_ids
    )

    return str(output_path)


def verify_onnx_model(
    onnx_path: str,
    pytorch_model,
    input_ids,
    attention_mask,
    token_type_ids
) -> bool:
    """验证 ONNX 模型与 PyTorch 输出一致性"""
    print("\nVerifying ONNX model...")

    # 检查 ONNX 模型结构
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  [✓] ONNX model structure check passed")

    # 打印模型输入输出信息
    print("\n  Model inputs:")
    for inp in onnx_model.graph.input:
        print(f"    - {inp.name}: {[d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]}")

    print("\n  Model outputs:")
    for out in onnx_model.graph.output:
        print(f"    - {out.name}: {[d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]}")

    if not HAS_ORT:
        print("\n  [!] Skipping output verification (onnxruntime not installed)")
        return True

    # 对比输出
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_outputs = pytorch_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    pytorch_logits = pytorch_outputs.logits.numpy()

    # ONNX Runtime 推理
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "token_type_ids": token_type_ids.numpy()
    }
    ort_outputs = ort_session.run(None, ort_inputs)

    # 比较输出
    max_diff = np.abs(pytorch_logits - ort_outputs[0]).max()
    mean_diff = np.abs(pytorch_logits - ort_outputs[0]).mean()

    print(f"\n  Output comparison:")
    print(f"    PyTorch logits: {pytorch_logits[0]}")
    print(f"    ONNX logits:    {ort_outputs[0][0]}")
    print(f"    Max difference: {max_diff:.6e}")
    print(f"    Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("\n  [✓] Verification PASSED!")
        return True
    elif max_diff < 1e-3:
        print(f"\n  [!] Warning: Outputs differ by {max_diff:.6e} (acceptable)")
        return True
    else:
        print(f"\n  [✗] Verification FAILED: Outputs differ by {max_diff:.6e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export Tiny-BERT model to ONNX format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="prajjwal1/bert-tiny",
        help="HuggingFace model name (default: prajjwal1/bert-tiny)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/bert_tiny.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)"
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=2,
        help="Number of classification labels (default: 2)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Tiny-BERT ONNX Export")
    print("=" * 60)
    print()

    export_bert_to_onnx(
        model_name=args.model,
        output_path=args.output,
        opset_version=args.opset,
        max_seq_length=args.max_seq_length,
        num_labels=args.num_labels,
        verbose=args.verbose
    )

    print()
    print("=" * 60)
    print("Export completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
