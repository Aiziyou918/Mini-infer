"""
Export Trained LeNet-5 Model

This script exports a trained PyTorch LeNet-5 model to either:
1. Binary weight files (for direct loading in Mini-Infer C++)
2. ONNX format (for ONNX Runtime or other frameworks)
3. Both formats

Usage:
    # Export weights as binary files (default)
    python export_lenet5.py --format weights
    
    # Export to ONNX
    python export_lenet5.py --format onnx --output ./models/lenet5.onnx
    
    # Export both formats
    python export_lenet5.py --format both
"""

import argparse
import json
from pathlib import Path

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np

from lenet5_model import LeNet5


def export_tensor(tensor, output_path):
    """
    Export PyTorch tensor to binary file
    
    Args:
        tensor (torch.Tensor): Tensor to export
        output_path (str): Output binary file path
    """
    # Convert to numpy and save as binary
    numpy_array = tensor.cpu().detach().numpy()
    numpy_array.astype(np.float32).tofile(output_path)
    
    # Print info
    file_size = Path(output_path).stat().st_size / 1024
    shape_str = str(tuple(numpy_array.shape))
    print(f"  Exported: {Path(output_path).name:30s} "
          f"shape={shape_str:20s} "
          f"size={file_size:8.2f} KB")
    
    return numpy_array.shape


def export_weights(model, output_dir):
    """
    Export all model weights to binary files
    
    Args:
        model (nn.Module): The model to export
        output_dir (str): Directory to save weight files
        
    Returns:
        dict: Metadata about exported weights
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Exporting Weights to Binary Files")
    print("=" * 70)
    
    model.eval()
    metadata = {
        'model': 'LeNet5',
        'format': 'float32',
        'endian': 'little',
        'layers': {}
    }
    
    # Export Conv1
    print("\nConv1 Layer:")
    conv1_weight_shape = export_tensor(
        model.conv1.weight.data, 
        output_dir / "conv1_weight.bin"
    )
    conv1_bias_shape = export_tensor(
        model.conv1.bias.data, 
        output_dir / "conv1_bias.bin"
    )
    metadata['layers']['conv1'] = {
        'type': 'Conv2d',
        'weight': {'shape': list(conv1_weight_shape), 'file': 'conv1_weight.bin'},
        'bias': {'shape': list(conv1_bias_shape), 'file': 'conv1_bias.bin'}
    }
    
    # Export Conv2
    print("\nConv2 Layer:")
    conv2_weight_shape = export_tensor(
        model.conv2.weight.data, 
        output_dir / "conv2_weight.bin"
    )
    conv2_bias_shape = export_tensor(
        model.conv2.bias.data, 
        output_dir / "conv2_bias.bin"
    )
    metadata['layers']['conv2'] = {
        'type': 'Conv2d',
        'weight': {'shape': list(conv2_weight_shape), 'file': 'conv2_weight.bin'},
        'bias': {'shape': list(conv2_bias_shape), 'file': 'conv2_bias.bin'}
    }
    
    # Export FC1
    print("\nFC1 Layer:")
    fc1_weight_shape = export_tensor(
        model.fc1.weight.data, 
        output_dir / "fc1_weight.bin"
    )
    fc1_bias_shape = export_tensor(
        model.fc1.bias.data, 
        output_dir / "fc1_bias.bin"
    )
    metadata['layers']['fc1'] = {
        'type': 'Linear',
        'weight': {'shape': list(fc1_weight_shape), 'file': 'fc1_weight.bin'},
        'bias': {'shape': list(fc1_bias_shape), 'file': 'fc1_bias.bin'}
    }
    
    # Export FC2
    print("\nFC2 Layer:")
    fc2_weight_shape = export_tensor(
        model.fc2.weight.data, 
        output_dir / "fc2_weight.bin"
    )
    fc2_bias_shape = export_tensor(
        model.fc2.bias.data, 
        output_dir / "fc2_bias.bin"
    )
    metadata['layers']['fc2'] = {
        'type': 'Linear',
        'weight': {'shape': list(fc2_weight_shape), 'file': 'fc2_weight.bin'},
        'bias': {'shape': list(fc2_bias_shape), 'file': 'fc2_bias.bin'}
    }
    
    # Export FC3
    print("\nFC3 Layer:")
    fc3_weight_shape = export_tensor(
        model.fc3.weight.data, 
        output_dir / "fc3_weight.bin"
    )
    fc3_bias_shape = export_tensor(
        model.fc3.bias.data, 
        output_dir / "fc3_bias.bin"
    )
    metadata['layers']['fc3'] = {
        'type': 'Linear',
        'weight': {'shape': list(fc3_weight_shape), 'file': 'fc3_weight.bin'},
        'bias': {'shape': list(fc3_bias_shape), 'file': 'fc3_bias.bin'}
    }
    
    # Save metadata
    metadata_path = output_dir / "weights_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Weights exported to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    print("=" * 70)
    
    # Calculate total size
    total_size = sum(
        f.stat().st_size 
        for f in output_dir.glob("*.bin")
    ) / 1024
    print(f"\nTotal weights size: {total_size:.2f} KB")
    print(f"Number of files: {len(list(output_dir.glob('*.bin')))}")
    
    return metadata


def export_to_onnx(model, output_path, opset_version=11, verbose=False):
    """Export PyTorch model to ONNX format"""
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    
    print(f"\nExporting model to ONNX format...")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Opset version: {opset_version}")
    
    torch.onnx.export(
        model, dummy_input, output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        verbose=verbose
    )
    
    print(f"Model exported to: {output_path}")


def verify_onnx_model(onnx_path, pytorch_model):
    """Verify ONNX model correctness"""
    print("\n" + "=" * 70)
    print("Verifying ONNX Model")
    print("=" * 70)
    
    # Load and check ONNX model
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed!")
    except Exception as e:
        print(f"ONNX model check failed: {e}")
        return False
    
    # Compare outputs
    test_input = torch.randn(1, 1, 28, 28)
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    ort_session = ort.InferenceSession(onnx_path)
    onnx_output = ort_session.run(None, {'input': test_input.numpy()})[0]
    
    max_diff = np.abs(pytorch_output - onnx_output).max()
    print(f"  Max difference: {max_diff:.6e}")
    
    if max_diff < 1e-5:
        print("\nVerification PASSED!")
        return True
    else:
        print(f"\nWarning: Outputs differ by {max_diff:.6e}")
        return max_diff < 1e-3


def main():
    parser = argparse.ArgumentParser(description='Export LeNet-5 Model')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/lenet5_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--format', type=str, default='weights', choices=['onnx', 'weights', 'both'],
                        help='Export format: onnx, weights (binary), or both')
    parser.add_argument('--output', type=str, default='./models/lenet5.onnx',
                        help='Output path for ONNX model')
    parser.add_argument('--weights-dir', type=str, default='./weights',
                        help='Directory to save weight binary files')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed export information')
    args = parser.parse_args()
    
    print("=" * 70)
    print("LeNet-5 Model Export Tool")
    print("=" * 70)
    print(f"Export format: {args.format.upper()}")
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\nError: Checkpoint not found: {checkpoint_path}")
        print("Please train the model first: python train_lenet5.py")
        return
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model = LeNet5(num_classes=10)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Test Accuracy: {checkpoint.get('test_accuracy', 'N/A'):.2f}%")
    
    # Export based on format
    if args.format in ['weights', 'both']:
        # Export weights to binary files
        weights_dir = Path(args.weights_dir)
        metadata = export_weights(model, weights_dir)
        
        print(f"\n{'='*70}")
        print("Weight Export Summary:")
        print(f"  Format: Binary (float32, little-endian)")
        print(f"  Location: {weights_dir}")
        print(f"  Total layers: {len(metadata['layers'])}")
        print(f"  Metadata: {weights_dir / 'weights_metadata.json'}")
        print(f"{'='*70}")
    
    if args.format in ['onnx', 'both']:
        # Export to ONNX
        export_to_onnx(model, str(output_path), args.opset, args.verbose)
        
        file_size = output_path.stat().st_size / 1024
        print(f"\nONNX Model size: {file_size:.2f} KB")
        
        # Verify
        verify_passed = verify_onnx_model(str(output_path), model)
        
        if not verify_passed:
            print("\nWarning: ONNX verification failed!")
    
    # Final summary
    print("\n" + "=" * 70)
    print("Export Completed Successfully!")
    print("=" * 70)
    
    if args.format in ['weights', 'both']:
        print(f"\nBinary weights ready for Mini-Infer:")
        print(f"  Directory: {Path(args.weights_dir).absolute()}")
        print(f"  Files: 10 weight files + 1 metadata file")
        
    if args.format in ['onnx', 'both']:
        print(f"\nONNX model ready for inference:")
        print(f"  File: {output_path.absolute()}")
    
    print("\nNext steps:")
    print("  1. Load weights in C++: see examples/lenet5_inference.cpp")
    print("  2. Run inference with Mini-Infer")
    print("=" * 70)


if __name__ == '__main__':
    main()
