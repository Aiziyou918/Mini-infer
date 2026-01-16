#!/usr/bin/env python3
"""
Export BERT Test Samples

导出预处理好的 tokenized 输入样本，供 C++ 推理测试使用

Usage:
    python export_bert_samples.py --output-dir ./test_samples
    python export_bert_samples.py --output-dir ./test_samples --num-samples 10
"""

import argparse
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

# 测试文本样本（情感分类：0=负面，1=正面）
TEST_TEXTS = [
    ("This movie is absolutely wonderful and I loved every minute of it!", 1),
    ("Terrible film, waste of time and money.", 0),
    ("The acting was superb and the story was engaging.", 1),
    ("I couldn't finish watching, it was so boring.", 0),
    ("A masterpiece of modern cinema!", 1),
    ("Disappointing and poorly executed.", 0),
    ("Highly recommend this to everyone!", 1),
    ("One of the worst movies I've ever seen.", 0),
    ("Beautiful cinematography and great performances.", 1),
    ("The plot made no sense at all.", 0),
    ("An incredible journey that touched my heart.", 1),
    ("Boring, predictable, and a complete waste of time.", 0),
    ("Outstanding performances by the entire cast.", 1),
    ("I want my two hours back. Awful movie.", 0),
    ("A delightful experience from start to finish.", 1),
]


def export_samples(
    model_name: str = "prajjwal1/bert-tiny",
    output_dir: str = "./test_samples",
    max_seq_length: int = 128,
    num_samples: int = 10
) -> dict:
    """
    导出 tokenized 测试样本

    Args:
        model_name: HuggingFace 模型名称（用于 tokenizer）
        output_dir: 输出目录
        max_seq_length: 最大序列长度
        num_samples: 导出样本数量

    Returns:
        样本元数据字典
    """
    output_dir = Path(output_dir)
    binary_dir = output_dir / "binary"
    binary_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    samples_metadata = []
    texts_to_export = TEST_TEXTS[:num_samples]

    print(f"\nExporting {len(texts_to_export)} samples...")
    print(f"Max sequence length: {max_seq_length}")
    print()

    for idx, (text, label) in enumerate(texts_to_export):
        print(f"Sample {idx:04d}: {text[:50]}{'...' if len(text) > 50 else ''}")

        # Tokenize
        encoding = tokenizer(
            text,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )

        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)
        token_type_ids = encoding.get(
            "token_type_ids",
            np.zeros_like(input_ids)
        ).astype(np.int64)

        # 保存为二进制文件
        base_name = f"sample_{idx:04d}"

        input_ids_path = binary_dir / f"{base_name}_input_ids.bin"
        attention_mask_path = binary_dir / f"{base_name}_attention_mask.bin"
        token_type_ids_path = binary_dir / f"{base_name}_token_type_ids.bin"

        input_ids.tofile(input_ids_path)
        attention_mask.tofile(attention_mask_path)
        token_type_ids.tofile(token_type_ids_path)

        # 计算实际 token 数量（非 padding）
        actual_length = int(attention_mask.sum())

        # 记录元数据
        samples_metadata.append({
            "index": idx,
            "text": text,
            "label": label,
            "label_name": "positive" if label == 1 else "negative",
            "seq_length": max_seq_length,
            "actual_length": actual_length,
            "files": {
                "input_ids": f"binary/{base_name}_input_ids.bin",
                "attention_mask": f"binary/{base_name}_attention_mask.bin",
                "token_type_ids": f"binary/{base_name}_token_type_ids.bin"
            },
            "shape": list(input_ids.shape),
            "dtype": "int64"
        })

    # 保存元数据
    metadata = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "total_samples": len(samples_metadata),
        "dtype": "int64",
        "label_mapping": {
            "0": "negative",
            "1": "positive"
        },
        "samples": samples_metadata
    }

    metadata_path = output_dir / "samples_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print(f"Exported {len(samples_metadata)} samples to {output_dir}")
    print(f"Metadata saved to: {metadata_path}")

    # 打印统计信息
    positive_count = sum(1 for s in samples_metadata if s["label"] == 1)
    negative_count = len(samples_metadata) - positive_count
    print(f"\nLabel distribution:")
    print(f"  Positive: {positive_count}")
    print(f"  Negative: {negative_count}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Export BERT test samples for C++ inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="prajjwal1/bert-tiny",
        help="HuggingFace model name for tokenizer"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_samples",
        help="Output directory for samples"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to export (default: 10)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BERT Test Samples Export")
    print("=" * 60)
    print()

    export_samples(
        model_name=args.model,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        num_samples=args.num_samples
    )

    print()
    print("=" * 60)
    print("Export completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
