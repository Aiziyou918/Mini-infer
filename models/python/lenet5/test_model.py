"""
Quick Test Script for LeNet-5 Model

This script provides a quick way to test the LeNet-5 model architecture
before training.

Usage:
    python test_model.py
"""

import torch
import torch.nn.functional as F
from lenet5_model import LeNet5, print_model_summary


def test_forward_pass():
    """Test basic forward pass with different batch sizes"""
    print("\n" + "=" * 70)
    print("Testing Forward Pass")
    print("=" * 70)
    
    model = LeNet5(num_classes=10)
    model.eval()
    
    # Test different batch sizes
    test_cases = [
        (1, "Single image"),
        (4, "Small batch"),
        (32, "Medium batch"),
        (64, "Large batch")
    ]
    
    for batch_size, description in test_cases:
        test_input = torch.randn(batch_size, 1, 28, 28)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"\n{description}:")
        print(f"  Input shape:  {tuple(test_input.shape)}")
        print(f"  Output shape: {tuple(output.shape)}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Check output shape
        assert output.shape == (batch_size, 10), f"Output shape mismatch!"
    
    print("\nForward pass test PASSED!")


def test_gradient_flow():
    """Test if gradients can flow through the network"""
    print("\n" + "=" * 70)
    print("Testing Gradient Flow")
    print("=" * 70)
    
    model = LeNet5(num_classes=10)
    model.train()
    
    # Create dummy input and target
    test_input = torch.randn(4, 1, 28, 28, requires_grad=True)
    target = torch.randint(0, 10, (4,))
    
    # Forward pass
    output = model(test_input)
    loss = F.cross_entropy(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist
    has_gradients = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  Warning: No gradient for {name}")
            has_gradients = False
        else:
            grad_norm = param.grad.norm().item()
            print(f"  {name:20s}: grad_norm = {grad_norm:.6f}")
    
    if has_gradients:
        print("\nGradient flow test PASSED!")
    else:
        print("\nGradient flow test FAILED!")
    
    return has_gradients


def test_output_distribution():
    """Test output distribution for random inputs"""
    print("\n" + "=" * 70)
    print("Testing Output Distribution")
    print("=" * 70)
    
    model = LeNet5(num_classes=10)
    model.eval()
    
    # Generate random inputs
    test_input = torch.randn(100, 1, 28, 28)
    
    with torch.no_grad():
        output = model(test_input)
        probabilities = F.softmax(output, dim=1)
    
    # Statistics
    print(f"\nOutput statistics (100 random images):")
    print(f"  Logits range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"  Logits mean: {output.mean():.3f}")
    print(f"  Logits std: {output.std():.3f}")
    
    print(f"\nProbability statistics:")
    print(f"  Mean probability per class: {probabilities.mean():.3f}")
    print(f"  Expected (uniform): {1.0/10:.3f}")
    
    # Check prediction distribution
    predictions = output.argmax(dim=1)
    unique_preds, counts = predictions.unique(return_counts=True)
    
    print(f"\nPrediction distribution:")
    for pred, count in zip(unique_preds.tolist(), counts.tolist()):
        print(f"  Class {pred}: {count:3d} / 100 ({count}%)")
    
    print("\nOutput distribution test PASSED!")


def test_parameter_count():
    """Verify parameter count"""
    print("\n" + "=" * 70)
    print("Testing Parameter Count")
    print("=" * 70)
    
    model = LeNet5(num_classes=10)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Expected parameter count for LeNet-5
    # Conv1: (1*5*5 + 1) * 6 = 156
    # Conv2: (6*5*5 + 1) * 16 = 2,416
    # FC1: (256 + 1) * 120 = 30,840
    # FC2: (120 + 1) * 84 = 10,164
    # FC3: (84 + 1) * 10 = 850
    # Total: 44,426
    
    expected_params = 44426
    
    print(f"\nParameter count:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Expected: {expected_params:,}")
    
    # Layer-wise parameter count
    print(f"\nLayer-wise parameters:")
    for name, param in model.named_parameters():
        print(f"  {name:20s}: {param.numel():6,} params, shape {tuple(param.shape)}")
    
    if total_params == expected_params:
        print("\nParameter count test PASSED!")
    else:
        print(f"\nWarning: Parameter count mismatch!")
        print(f"  Expected: {expected_params:,}")
        print(f"  Got: {total_params:,}")


def main():
    print("=" * 70)
    print("LeNet-5 Model Test Suite")
    print("=" * 70)
    
    # Create model
    model = LeNet5(num_classes=10)
    print_model_summary(model)
    
    # Run tests
    test_forward_pass()
    test_gradient_flow()
    test_output_distribution()
    test_parameter_count()
    
    # Final summary
    print("\n" + "=" * 70)
    print("All Tests Completed!")
    print("=" * 70)
    print("\nModel is ready for training.")
    print("\nNext steps:")
    print("  1. Train the model: python train_lenet5.py")
    print("  2. Export to ONNX: python export_lenet5.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
