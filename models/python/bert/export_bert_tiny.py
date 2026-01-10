#!/usr/bin/env python3
"""
Export BERT-tiny model to ONNX format for Mini-Infer testing.

BERT-tiny configuration:
- L=2 (num_hidden_layers)
- H=128 (hidden_size)
- A=2 (num_attention_heads)
- ~4.4M parameters

Usage:
    pip install transformers torch onnx onnxruntime
    python export_bert_tiny.py
"""

import os
import torch
import numpy as np
from pathlib import Path

try:
    from transformers import BertModel, BertConfig, BertTokenizer
    from transformers.onnx import export
except ImportError:
    print("Please install transformers: pip install transformers torch")
    exit(1)


def create_bert_tiny_config():
    """Create BERT-tiny configuration."""
    return BertConfig(
        vocab_size=30522,           # Standard BERT vocabulary
        hidden_size=128,            # H=128
        num_hidden_layers=2,        # L=2
        num_attention_heads=2,      # A=2
        intermediate_size=512,      # 4 * H
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    )


def export_bert_tiny_onnx(output_dir: str = "."):
    """Export BERT-tiny model to ONNX format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Creating BERT-tiny model...")
    config = create_bert_tiny_config()
    model = BertModel(config)
    model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num layers: {config.num_hidden_layers}")
    print(f"Num attention heads: {config.num_attention_heads}")

    # Create dummy inputs
    batch_size = 1
    seq_length = 32

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)

    # Export to ONNX
    onnx_path = output_path / "bert_tiny.onnx"
    print(f"\nExporting to {onnx_path}...")

    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            "pooler_output": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    print(f"ONNX model saved to: {onnx_path}")
    print(f"File size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Generate test data
    print("\nGenerating test data...")
    test_data_dir = output_path / "test_samples"
    test_data_dir.mkdir(exist_ok=True)

    # Save input tensors
    input_ids_np = input_ids.numpy().astype(np.int64)
    attention_mask_np = attention_mask.numpy().astype(np.int64)
    token_type_ids_np = token_type_ids.numpy().astype(np.int64)

    input_ids_np.tofile(test_data_dir / "input_ids.bin")
    attention_mask_np.tofile(test_data_dir / "attention_mask.bin")
    token_type_ids_np.tofile(test_data_dir / "token_type_ids.bin")

    # Run inference and save outputs
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)

    last_hidden_state = outputs.last_hidden_state.numpy()
    pooler_output = outputs.pooler_output.numpy()

    last_hidden_state.tofile(test_data_dir / "last_hidden_state.bin")
    pooler_output.tofile(test_data_dir / "pooler_output.bin")

    # Save metadata
    metadata = {
        "batch_size": batch_size,
        "seq_length": seq_length,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "input_ids_shape": list(input_ids_np.shape),
        "last_hidden_state_shape": list(last_hidden_state.shape),
        "pooler_output_shape": list(pooler_output.shape),
    }

    import json
    with open(test_data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Test data saved to: {test_data_dir}")
    print(f"  - input_ids.bin: {input_ids_np.shape}")
    print(f"  - attention_mask.bin: {attention_mask_np.shape}")
    print(f"  - token_type_ids.bin: {token_type_ids_np.shape}")
    print(f"  - last_hidden_state.bin: {last_hidden_state.shape}")
    print(f"  - pooler_output.bin: {pooler_output.shape}")

    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort
        print("\nVerifying with ONNX Runtime...")

        session = ort.InferenceSession(str(onnx_path))
        ort_outputs = session.run(
            None,
            {
                "input_ids": input_ids_np,
                "attention_mask": attention_mask_np,
                "token_type_ids": token_type_ids_np,
            }
        )

        # Compare outputs
        torch_lhs = last_hidden_state
        ort_lhs = ort_outputs[0]

        max_diff = np.abs(torch_lhs - ort_lhs).max()
        mean_diff = np.abs(torch_lhs - ort_lhs).mean()

        print(f"PyTorch vs ONNX Runtime comparison:")
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")

        if max_diff < 1e-4:
            print("  [PASS] ONNX export verified!")
        else:
            print("  [WARN] Large difference detected")

    except ImportError:
        print("\nSkipping ONNX Runtime verification (onnxruntime not installed)")

    print("\nDone!")
    return str(onnx_path)


def export_bert_tiny_simplified(output_dir: str = "."):
    """
    Export a simplified BERT-tiny model without dynamic axes.
    This is easier for Mini-Infer to handle initially.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Creating simplified BERT-tiny model...")
    config = create_bert_tiny_config()
    model = BertModel(config)
    model.eval()

    # Fixed input shape
    batch_size = 1
    seq_length = 32

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)

    # Export to ONNX with fixed shapes
    onnx_path = output_path / "bert_tiny_fixed.onnx"
    print(f"\nExporting fixed-shape model to {onnx_path}...")

    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        opset_version=14,
        do_constant_folding=True,
    )

    print(f"Fixed-shape ONNX model saved to: {onnx_path}")
    print(f"File size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")

    return str(onnx_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export BERT-tiny to ONNX")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory")
    parser.add_argument("--simplified", "-s", action="store_true",
                        help="Export simplified model with fixed shapes")
    args = parser.parse_args()

    if args.simplified:
        export_bert_tiny_simplified(args.output_dir)
    else:
        export_bert_tiny_onnx(args.output_dir)
