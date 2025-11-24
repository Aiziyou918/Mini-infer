"""
Verify Exported Binary Weights

This script verifies that the exported binary weight files match
the original PyTorch model weights.

Usage:
    python verify_weights.py --checkpoint ./checkpoints/lenet5_best.pth \
                            --weights-dir ./weights
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from lenet5_model import LeNet5


def load_binary_weight(filepath, shape):
    """Load binary weight file and reshape"""
    data = np.fromfile(filepath, dtype=np.float32)
    expected_size = np.prod(shape)
    
    if len(data) != expected_size:
        raise ValueError(
            f"Size mismatch for {filepath}: "
            f"got {len(data)}, expected {expected_size}"
        )
    
    return data.reshape(shape)


def verify_layer_weights(layer_name, pytorch_weight, pytorch_bias, weights_dir):
    """Verify weights and bias for a single layer"""
    print(f"\n{layer_name}:")
    
    # Load binary files
    weight_file = weights_dir / f"{layer_name}_weight.bin"
    bias_file = weights_dir / f"{layer_name}_bias.bin"
    
    if not weight_file.exists():
        print(f"  ERROR: Weight file not found: {weight_file}")
        return False
    
    if not bias_file.exists():
        print(f"  ERROR: Bias file not found: {bias_file}")
        return False
    
    # Load and compare weights
    binary_weight = load_binary_weight(weight_file, pytorch_weight.shape)
    weight_diff = np.abs(binary_weight - pytorch_weight).max()
    
    print(f"  Weight shape: {pytorch_weight.shape}")
    print(f"  Weight max diff: {weight_diff:.6e}")
    
    # Load and compare bias
    binary_bias = load_binary_weight(bias_file, pytorch_bias.shape)
    bias_diff = np.abs(binary_bias - pytorch_bias).max()
    
    print(f"  Bias shape: {pytorch_bias.shape}")
    print(f"  Bias max diff: {bias_diff:.6e}")
    
    # Check threshold
    threshold = 1e-6
    weight_ok = weight_diff < threshold
    bias_ok = bias_diff < threshold
    
    if weight_ok and bias_ok:
        print(f"  [SUCCESS] PASSED")
        return True
    else:
        if not weight_ok:
            print(f"  [FAILED] FAILED: Weight diff {weight_diff} > {threshold}")
        if not bias_ok:
            print(f"  [FAILED] FAILED: Bias diff {bias_diff} > {threshold}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Verify exported weights')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/lenet5_best.pth',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--weights-dir', type=str,
                        default='./weights',
                        help='Directory containing binary weight files')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Binary Weight Verification")
    print("=" * 70)
    
    # Load PyTorch model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\nError: Checkpoint not found: {checkpoint_path}")
        return 1
    
    print(f"\nLoading PyTorch checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model = LeNet5(num_classes=10)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Test Accuracy: {checkpoint.get('test_accuracy', 'N/A'):.2f}%")
    
    # Verify weights directory
    weights_dir = Path(args.weights_dir)
    if not weights_dir.exists():
        print(f"\nError: Weights directory not found: {weights_dir}")
        print("Please export weights first: python export_lenet5.py --format weights")
        return 1
    
    print(f"\nBinary weights directory: {weights_dir}")
    
    # Verify each layer
    print("\n" + "=" * 70)
    print("Verifying Layers")
    print("=" * 70)
    
    all_passed = True
    
    # Conv1
    all_passed &= verify_layer_weights(
        'conv1',
        model.conv1.weight.data.numpy(),
        model.conv1.bias.data.numpy(),
        weights_dir
    )
    
    # Conv2
    all_passed &= verify_layer_weights(
        'conv2',
        model.conv2.weight.data.numpy(),
        model.conv2.bias.data.numpy(),
        weights_dir
    )
    
    # FC1
    all_passed &= verify_layer_weights(
        'fc1',
        model.fc1.weight.data.numpy(),
        model.fc1.bias.data.numpy(),
        weights_dir
    )
    
    # FC2
    all_passed &= verify_layer_weights(
        'fc2',
        model.fc2.weight.data.numpy(),
        model.fc2.bias.data.numpy(),
        weights_dir
    )
    
    # FC3
    all_passed &= verify_layer_weights(
        'fc3',
        model.fc3.weight.data.numpy(),
        model.fc3.bias.data.numpy(),
        weights_dir
    )
    
    # File size summary
    print("\n" + "=" * 70)
    print("File Size Summary")
    print("=" * 70)
    
    total_size = 0
    for bin_file in sorted(weights_dir.glob("*.bin")):
        size_kb = bin_file.stat().st_size / 1024
        total_size += size_kb
        print(f"  {bin_file.name:30s} {size_kb:8.2f} KB")
    
    print(f"  {'Total:':30s} {total_size:8.2f} KB")
    
    # Check metadata file
    metadata_file = weights_dir / "weights_metadata.json"
    if metadata_file.exists():
        print(f"\n  [SUCCESS] Metadata file found: {metadata_file.name}")
    else:
        print(f"\n  [FAILED] Warning: Metadata file not found")
        all_passed = False
    
    # Final result
    print("\n" + "=" * 70)
    if all_passed:
        print("[SUCCESS] ALL CHECKS PASSED!")
        print("Binary weights are correct and ready to use.")
    else:
        print("[FAILED] VERIFICATION FAILED!")
        print("Please re-export weights: python export_lenet5.py --format weights")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
