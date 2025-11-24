"""
LeNet-5 Model Definition for MNIST

Classic LeNet-5 architecture (LeCun et al., 1998) adapted for MNIST.
This model will be exported to ONNX format for Mini-Infer inference engine.

Architecture:
    Input: 1x28x28 (grayscale MNIST images)
    Conv1: 6 filters, 5x5 kernel -> 6x24x24
    MaxPool1: 2x2 -> 6x12x12
    Conv2: 16 filters, 5x5 kernel -> 16x8x8
    MaxPool2: 2x2 -> 16x4x4
    Flatten: 256
    FC1: 256 -> 120
    FC2: 120 -> 84
    FC3: 84 -> 10 (output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet-5 Convolutional Neural Network
    
    Original paper: "Gradient-Based Learning Applied to Document Recognition"
    by Yann LeCun et al., 1998
    """
    
    def __init__(self, num_classes=10):
        """
        Initialize LeNet-5 model
        
        Args:
            num_classes (int): Number of output classes (default: 10 for MNIST)
        """
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1,      # Grayscale input
            out_channels=6,     # 6 feature maps
            kernel_size=5,      # 5x5 kernel
            stride=1,
            padding=0           # No padding: 28x28 -> 24x24
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0           # No padding: 12x12 -> 8x8
        )
        
        # Pooling layer (used after both conv layers)
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        
        # Fully connected layers
        # After conv2 + pool2: 16 x 4 x 4 = 256
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 1, 28, 28)
            
        Returns:
            torch.Tensor: Output logits of shape (N, num_classes)
        """
        # Conv1 -> ReLU -> MaxPool
        # 1x28x28 -> 6x24x24 -> 6x12x12
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv2 -> ReLU -> MaxPool
        # 6x12x12 -> 16x8x8 -> 16x4x4
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten: 16x4x4 -> 256
        x = x.view(-1, 16 * 4 * 4)
        
        # FC1 -> ReLU
        x = F.relu(self.fc1(x))
        
        # FC2 -> ReLU
        x = F.relu(self.fc2(x))
        
        # FC3 (output layer, no activation)
        x = self.fc3(x)
        
        return x
    
    def get_layer_info(self):
        """
        Get detailed information about each layer
        
        Returns:
            dict: Layer information for debugging and visualization
        """
        info = {
            'conv1': {
                'type': 'Conv2d',
                'params': f'in=1, out=6, kernel=5x5, stride=1, padding=0',
                'output_shape': '(N, 6, 24, 24)'
            },
            'pool1': {
                'type': 'MaxPool2d',
                'params': 'kernel=2x2, stride=2',
                'output_shape': '(N, 6, 12, 12)'
            },
            'conv2': {
                'type': 'Conv2d',
                'params': 'in=6, out=16, kernel=5x5, stride=1, padding=0',
                'output_shape': '(N, 16, 8, 8)'
            },
            'pool2': {
                'type': 'MaxPool2d',
                'params': 'kernel=2x2, stride=2',
                'output_shape': '(N, 16, 4, 4)'
            },
            'fc1': {
                'type': 'Linear',
                'params': 'in=256, out=120',
                'output_shape': '(N, 120)'
            },
            'fc2': {
                'type': 'Linear',
                'params': 'in=120, out=84',
                'output_shape': '(N, 84)'
            },
            'fc3': {
                'type': 'Linear',
                'params': 'in=84, out=10',
                'output_shape': '(N, 10)'
            }
        }
        return info


def print_model_summary(model, input_size=(1, 1, 28, 28)):
    """
    Print model summary including layer details and parameter count
    
    Args:
        model (nn.Module): The model to summarize
        input_size (tuple): Input tensor size for testing
    """
    print("=" * 70)
    print("LeNet-5 Model Summary")
    print("=" * 70)
    
    # Layer information
    layer_info = model.get_layer_info()
    print("\nLayer Architecture:")
    print("-" * 70)
    for layer_name, info in layer_info.items():
        print(f"{layer_name:10s} | {info['type']:12s} | {info['params']:40s}")
        print(f"           | Output: {info['output_shape']}")
        print("-" * 70)
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"\nForward pass test:")
        print(f"  Input shape:  {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print("\nModel is ready for training!")
    except Exception as e:
        print(f"\nWarning: Forward pass failed: {e}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Test model creation
    model = LeNet5(num_classes=10)
    print_model_summary(model)
    
    # Test with a batch
    print("\nTesting with batch size 4:")
    batch_input = torch.randn(4, 1, 28, 28)
    output = model(batch_input)
    print(f"Batch input shape: {batch_input.shape}")
    print(f"Batch output shape: {output.shape}")
    print(f"Output logits range: [{output.min().item():.3f}, {output.max().item():.3f}]")
