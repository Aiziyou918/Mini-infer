#!/usr/bin/env python3
"""
Generate PyTorch Reference Outputs for BERT

生成 PyTorch 推理的参考输出，用于与 Mini-Infer 结果对比

Usage:
    python generate_reference_outputs.py
    python generate_reference_outputs.py --samples-dir ./test_samples --model prajjwal1/bert-tiny
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification


def generate_reference_outputs(
    model_name: str = "prajjwal1/bert-tiny",
    samples_dir: str = "./test_samples",
    output_path: str = None,
    num_labels: int = 2,
    weights_path: str = None
) -> dict:
    """
    生成 PyTorch 参考输出

    Args:
        model_name: HuggingFace 模型名称
        samples_dir: 测试样本目录
        output_path: 输出文件路径
        num_labels: 分类标签数量
        weights_path: 预保存的模型权重路径（确保与 ONNX 一致）

    Returns:
        参考输出字典
    """
    samples_dir = Path(samples_dir)

    if output_path is None:
        output_path = samples_dir / "reference_outputs.json"
    else:
        output_path = Path(output_path)

    # 加载元数据
    metadata_path = samples_dir / "samples_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 加载模型
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # 如果提供了权重路径，加载预保存的权重（确保与 ONNX 一致）
    if weights_path is None:
        # 默认查找 models 目录下的权重文件
        default_weights = samples_dir.parent / "models" / "bert_tiny_weights.pt"
        if default_weights.exists():
            weights_path = default_weights

    if weights_path and Path(weights_path).exists():
        print(f"Loading saved weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("Weights loaded successfully!")
    else:
        print("[WARNING] No saved weights found, using randomly initialized classifier!")
        print("         Results may not match ONNX model output.")

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")

    results = []
    correct_count = 0

    print(f"\nGenerating reference outputs for {len(metadata['samples'])} samples...")
    print()

    for sample in metadata["samples"]:
        idx = sample["index"]
        shape = sample["shape"]

        # 加载输入
        input_ids = np.fromfile(
            samples_dir / sample["files"]["input_ids"],
            dtype=np.int64
        ).reshape(shape)

        attention_mask = np.fromfile(
            samples_dir / sample["files"]["attention_mask"],
            dtype=np.int64
        ).reshape(shape)

        token_type_ids = np.fromfile(
            samples_dir / sample["files"]["token_type_ids"],
            dtype=np.int64
        ).reshape(shape)

        # PyTorch 推理
        with torch.no_grad():
            outputs = model(
                torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention_mask),
                token_type_ids=torch.from_numpy(token_type_ids)
            )

        logits = outputs.logits.numpy()
        probs = torch.softmax(outputs.logits, dim=-1).numpy()
        predicted = int(np.argmax(logits[0]))

        is_correct = predicted == sample["label"]
        if is_correct:
            correct_count += 1

        result = {
            "index": idx,
            "text": sample["text"],
            "label": sample["label"],
            "label_name": sample.get("label_name", ""),
            "logits": logits[0].tolist(),
            "probabilities": probs[0].tolist(),
            "predicted": predicted,
            "predicted_name": "positive" if predicted == 1 else "negative",
            "correct": is_correct
        }
        results.append(result)

        status = "✓" if is_correct else "✗"
        print(f"Sample {idx:04d}: pred={predicted} ({result['predicted_name']:8s}), "
              f"label={sample['label']} ({sample.get('label_name', ''):8s}) [{status}]")

    # 计算准确率
    accuracy = correct_count / len(results) * 100 if results else 0

    # 保存结果
    output_data = {
        "model_name": model_name,
        "total_samples": len(results),
        "correct": correct_count,
        "accuracy": accuracy,
        "label_mapping": metadata.get("label_mapping", {}),
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print()
    print(f"Reference outputs saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total samples: {len(results)}")
    print(f"  Correct: {correct_count}")
    print(f"  Accuracy: {accuracy:.2f}%")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyTorch reference outputs for BERT"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="prajjwal1/bert-tiny",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="./test_samples",
        help="Directory containing test samples"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: <samples_dir>/reference_outputs.json)"
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=2,
        help="Number of classification labels (default: 2)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to saved model weights (default: auto-detect from models/bert_tiny_weights.pt)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BERT Reference Output Generation")
    print("=" * 60)
    print()

    generate_reference_outputs(
        model_name=args.model,
        samples_dir=args.samples_dir,
        output_path=args.output,
        num_labels=args.num_labels,
        weights_path=args.weights
    )

    print()
    print("=" * 60)
    print("Generation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
