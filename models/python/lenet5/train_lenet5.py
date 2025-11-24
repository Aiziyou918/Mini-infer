"""
Train LeNet-5 on MNIST Dataset

This script trains a LeNet-5 model on the MNIST dataset and saves the trained
model for later export to ONNX format for Mini-Infer inference engine.

Usage:
    python train_lenet5.py [--epochs 10] [--batch-size 64] [--lr 0.001]
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lenet5_model import LeNet5, print_model_summary


def get_mnist_loaders(batch_size=64, data_dir='./data'):
    """
    Create MNIST data loaders for training and testing
    
    Args:
        batch_size (int): Batch size for training
        data_dir (str): Directory to store/load MNIST data
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),                          # Convert to tensor [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))     # MNIST mean and std
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader


def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch
    
    Args:
        model (nn.Module): The model to train
        device (torch.device): Device to train on
        train_loader (DataLoader): Training data loader
        optimizer (Optimizer): Optimizer
        criterion (Loss): Loss function
        epoch (int): Current epoch number
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    elapsed = time.time() - start_time
    
    print(f'\nEpoch {epoch} Training Summary:')
    print(f'  Average Loss: {avg_loss:.4f}')
    print(f'  Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    print(f'  Time: {elapsed:.2f}s')
    
    return avg_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    """
    Evaluate the model on test set
    
    Args:
        model (nn.Module): The model to evaluate
        device (torch.device): Device to evaluate on
        test_loader (DataLoader): Test data loader
        criterion (Loss): Loss function
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest Set:')
    print(f'  Average Loss: {test_loss:.4f}')
    print(f'  Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    return test_loss, accuracy


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train LeNet-5 on MNIST')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='directory to save model checkpoints')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='directory to store MNIST data')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )
    
    # Create model
    print("\nCreating LeNet-5 model...")
    model = LeNet5(num_classes=10).to(device)
    print_model_summary(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    best_accuracy = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            checkpoint_path = save_dir / 'lenet5_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_acc,
                'test_loss': test_loss,
            }, checkpoint_path)
            print(f"\nSaved best model to {checkpoint_path}")
        
        # Save latest model
        checkpoint_path = save_dir / 'lenet5_latest.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'history': history
        }, checkpoint_path)
    
    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"\nCheckpoints saved to: {save_dir}")
    print(f"  - Best model: lenet5_best.pth")
    print(f"  - Latest model: lenet5_latest.pth")
    print("\nNext steps:")
    print("  1. Export model to ONNX: python export_lenet5.py")
    print("  2. Run inference with Mini-Infer")
    print("=" * 70)


if __name__ == '__main__':
    main()
