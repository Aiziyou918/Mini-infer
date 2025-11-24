#!/usr/bin/env python3
"""
Generate reference outputs from PyTorch model for testing
"""

import torch
import numpy as np
import json
from pathlib import Path
import argparse
from lenet5_model import LeNet5


def generate_reference_outputs(
    checkpoint_path: str,
    samples_dir: str,
    output_file: str
):
    """
    Generate reference outputs from PyTorch model
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        samples_dir: Directory containing test samples
        output_file: Output JSON file path
    """
    print("=" * 70)
    print("Generating Reference Outputs from PyTorch Model")
    print("=" * 70)
    print()
    
    # Load model
    print("Loading PyTorch model...")
    device = torch.device('cpu')
    model = LeNet5().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Model loaded from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Test accuracy: {checkpoint.get('test_accuracy', 'unknown'):.2f}%")
    print()
    
    # Find all binary samples
    samples_dir = Path(samples_dir)
    binary_dir = samples_dir / "binary"
    
    if not binary_dir.exists():
        binary_dir = samples_dir
    
    sample_files = sorted(binary_dir.glob("*.bin"))
    
    if not sample_files:
        raise FileNotFoundError(f"No .bin files found in {binary_dir}")
    
    print(f"Found {len(sample_files)} test samples in {binary_dir}")
    print()
    
    # Generate outputs
    print("Generating reference outputs...")
    reference_outputs = []
    
    with torch.no_grad():
        for i, sample_file in enumerate(sample_files):
            # Load sample (already normalized)
            data = np.fromfile(sample_file, dtype=np.float32)
            data = data.reshape(1, 1, 28, 28)
            
            # Convert to tensor
            input_tensor = torch.from_numpy(data).to(device)
            
            # Forward pass
            output = model(input_tensor)
            
            # Get logits and probabilities
            logits = output.cpu().numpy()[0].tolist()  # Shape: [10]
            probs = torch.softmax(output, dim=1).cpu().numpy()[0].tolist()
            predicted = int(output.argmax(dim=1).item())
            
            # Extract label from filename
            filename = sample_file.stem
            label = -1
            if "_label_" in filename:
                try:
                    label = int(filename.split("_label_")[1])
                except:
                    pass
            
            # Store result
            reference_outputs.append({
                "index": i,
                "filename": sample_file.name,
                "label": label,
                "predicted": predicted,
                "logits": logits,
                "probabilities": probs,
                "confidence": float(max(probs)),
                "correct": predicted == label if label != -1 else None
            })
            
            if (i + 1) % 10 == 0 or i == len(sample_files) - 1:
                print(f"  Processed {i + 1}/{len(sample_files)} samples")
    
    # Calculate statistics
    correct_count = sum(1 for r in reference_outputs if r.get("correct") == True)
    total_count = sum(1 for r in reference_outputs if r.get("correct") is not None)
    accuracy = 100.0 * correct_count / total_count if total_count > 0 else 0.0
    
    print()
    print("PyTorch Model Results:")
    print(f"  Accuracy: {correct_count}/{total_count} ({accuracy:.2f}%)")
    print(f"  Average confidence: {np.mean([r['confidence'] for r in reference_outputs]):.4f}")
    print()
    
    # Save to JSON
    output_data = {
        "model_checkpoint": str(checkpoint_path),
        "samples_directory": str(binary_dir),
        "total_samples": len(reference_outputs),
        "accuracy": accuracy,
        "results": reference_outputs
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Reference outputs saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
    print()
    print("=" * 70)
    print("[SUCCESS] Reference outputs generated successfully!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Generate reference outputs from PyTorch model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./checkpoints/lenet5_best.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--samples-dir',
        type=str,
        default='./test_samples',
        help='Directory containing test samples'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./test_samples/reference_outputs.json',
        help='Output JSON file'
    )
    
    args = parser.parse_args()
    
    generate_reference_outputs(
        args.checkpoint,
        args.samples_dir,
        args.output
    )


if __name__ == '__main__':
    main()
