"""
Export MNIST Test Samples for Inference

This script exports a subset of MNIST test samples in formats suitable for
C++ inference testing with Mini-Infer.

Supports multiple export formats:
- Binary (.bin): Raw float32 data for direct loading
- PNG images: Visual inspection and debugging
- Normalized arrays: Preprocessed and ready for inference

Usage:
    # Export 10 samples (one per class)
    python export_mnist_samples.py --num-samples 10 --output-dir ./test_samples
    
    # Export 100 random samples
    python export_mnist_samples.py --num-samples 100 --random
    
    # Export specific classes only
    python export_mnist_samples.py --classes 0 1 2 --num-per-class 5
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms


def load_mnist_test_set(data_dir='./data'):
    """Load MNIST test dataset"""
    # Original dataset (uint8, 0-255)
    dataset_raw = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=None
    )
    
    # Normalized dataset (float32, preprocessed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_normalized = datasets.MNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=transform
    )
    
    return dataset_raw, dataset_normalized


def select_samples(dataset, num_samples=10, classes=None, num_per_class=None, random_seed=42):
    """
    Select samples from dataset
    
    Args:
        dataset: MNIST dataset
        num_samples: Total number of samples (if num_per_class not specified)
        classes: List of classes to include (default: all 0-9)
        num_per_class: Number of samples per class (overrides num_samples)
        random_seed: Random seed for reproducibility
        
    Returns:
        List of (index, label) tuples
    """
    np.random.seed(random_seed)
    
    if classes is None:
        classes = list(range(10))
    
    # Group indices by class
    class_indices = {c: [] for c in classes}
    for idx, (_, label) in enumerate(dataset):
        if label in classes:
            class_indices[label].append(idx)
    
    # Select samples
    selected = []
    
    if num_per_class is not None:
        # Select num_per_class from each class
        for cls in classes:
            indices = class_indices[cls]
            if len(indices) >= num_per_class:
                chosen = np.random.choice(indices, num_per_class, replace=False)
            else:
                chosen = indices
            selected.extend([(idx, cls) for idx in chosen])
    else:
        # Select num_samples total, balanced across classes
        samples_per_class = num_samples // len(classes)
        remainder = num_samples % len(classes)
        
        for i, cls in enumerate(classes):
            indices = class_indices[cls]
            n = samples_per_class + (1 if i < remainder else 0)
            if len(indices) >= n:
                chosen = np.random.choice(indices, n, replace=False)
            else:
                chosen = indices
            selected.extend([(idx, cls) for idx in chosen])
    
    # Shuffle
    np.random.shuffle(selected)
    
    return selected


def export_sample(raw_img, norm_tensor, label, idx, output_dir, formats):
    """
    Export a single sample in specified formats
    
    Args:
        raw_img: PIL Image (uint8, 28x28)
        norm_tensor: Normalized tensor (float32, 1x28x28)
        label: Class label
        idx: Sample index for filename
        output_dir: Output directory
        formats: List of formats to export
        
    Returns:
        dict: Metadata about the exported sample
    """
    metadata = {
        'index': idx,
        'label': int(label),
        'shape': [1, 28, 28],
        'files': {}
    }
    
    # Export as PNG
    if 'png' in formats:
        png_path = output_dir / 'images' / f'sample_{idx:04d}_label_{label}.png'
        png_path.parent.mkdir(parents=True, exist_ok=True)
        raw_img.save(png_path)
        metadata['files']['png'] = str(png_path.relative_to(output_dir))
    
    # Export normalized tensor as binary
    if 'binary' in formats:
        bin_path = output_dir / 'binary' / f'sample_{idx:04d}.bin'
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy and save
        array = norm_tensor.numpy().astype(np.float32)
        array.tofile(bin_path)
        
        metadata['files']['binary'] = str(bin_path.relative_to(output_dir))
        metadata['binary_info'] = {
            'dtype': 'float32',
            'shape': list(array.shape),
            'size_bytes': array.nbytes,
            'mean': float(array.mean()),
            'std': float(array.std()),
            'min': float(array.min()),
            'max': float(array.max())
        }
    
    # Export raw tensor as numpy
    if 'npy' in formats:
        npy_path = output_dir / 'numpy' / f'sample_{idx:04d}.npy'
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, norm_tensor.numpy())
        metadata['files']['npy'] = str(npy_path.relative_to(output_dir))
    
    return metadata


def create_cpp_loader_template(samples_metadata, output_dir):
    """Generate C++ code template for loading samples"""
    
    cpp_code = '''/**
 * MNIST Sample Loader for Mini-Infer
 * 
 * Auto-generated by export_mnist_samples.py
 */

#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include "mini_infer/core/tensor.h"

namespace mnist_loader {

/**
 * Load a single MNIST sample from binary file
 * 
 * @param filepath Path to .bin file
 * @return Tensor with shape [1, 1, 28, 28]
 */
std::shared_ptr<mini_infer::core::Tensor> load_sample(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open: " + filepath);
    }
    
    // Read data
    std::vector<float> data(1 * 28 * 28);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    
    if (!file) {
        throw std::runtime_error("Failed to read data from: " + filepath);
    }
    
    // Create tensor
    auto tensor = mini_infer::core::Tensor::create(
        mini_infer::core::Shape({1, 1, 28, 28}),
        mini_infer::core::DataType::FLOAT32
    );
    
    // Copy data
    std::memcpy(tensor->data(), data.data(), data.size() * sizeof(float));
    
    return tensor;
}

/**
 * Sample metadata
 */
struct SampleInfo {
    int index;
    int label;
    std::string binary_path;
    std::string png_path;
};

/**
 * Get all test samples
 */
std::vector<SampleInfo> get_test_samples() {
    return {
'''
    
    # Add sample info
    for sample in samples_metadata:
        idx = sample['index']
        label = sample['label']
        bin_path = sample['files'].get('binary', '')
        png_path = sample['files'].get('png', '')
        
        cpp_code += f'        {{{idx}, {label}, "{bin_path}", "{png_path}"}},\n'
    
    cpp_code += '''    };
}

/**
 * Print sample statistics
 */
void print_sample_info(const SampleInfo& info) {
    std::cout << "Sample " << info.index 
              << " (Label: " << info.label << ")" << std::endl;
    std::cout << "  Binary: " << info.binary_path << std::endl;
    std::cout << "  Image:  " << info.png_path << std::endl;
}

} // namespace mnist_loader

// Example usage:
/*
#include "mnist_loader.h"

int main() {
    auto samples = mnist_loader::get_test_samples();
    
    for (const auto& sample_info : samples) {
        // Load sample
        auto input = mnist_loader::load_sample(sample_info.binary_path);
        
        // Run inference
        auto output = model->forward({input});
        
        // Get prediction
        int predicted = argmax(output[0]);
        int ground_truth = sample_info.label;
        
        std::cout << "Sample " << sample_info.index 
                  << ": predicted=" << predicted 
                  << ", ground_truth=" << ground_truth 
                  << (predicted == ground_truth ? " [SUCCESS]" : " [FAILED]") 
                  << std::endl;
    }
}
*/
'''
    
    # Save to file
    cpp_path = output_dir / 'mnist_loader.h'
    with open(cpp_path, 'w', encoding='utf-8') as f:
        f.write(cpp_code)
    
    return cpp_path


def main():
    parser = argparse.ArgumentParser(
        description='Export MNIST test samples for C++ inference'
    )
    parser.add_argument('--output-dir', type=str, default='./test_samples',
                        help='Output directory for samples')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing MNIST data')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Total number of samples to export')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='Specific classes to export (default: all 0-9)')
    parser.add_argument('--num-per-class', type=int, default=None,
                        help='Number of samples per class (overrides num-samples)')
    parser.add_argument('--formats', nargs='+', 
                        default=['binary', 'png'],
                        choices=['binary', 'png', 'npy'],
                        help='Export formats')
    parser.add_argument('--random', action='store_true',
                        help='Select random samples (default: balanced per class)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MNIST Test Sample Export Tool")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"Export formats: {', '.join(args.formats)}")
    
    # Load MNIST
    print("\nLoading MNIST test dataset...")
    dataset_raw, dataset_normalized = load_mnist_test_set(args.data_dir)
    print(f"  Total test samples: {len(dataset_raw)}")
    
    # Select samples
    print("\nSelecting samples...")
    if args.num_per_class:
        print(f"  Samples per class: {args.num_per_class}")
        if args.classes:
            print(f"  Classes: {args.classes}")
        else:
            print(f"  Classes: 0-9 (all)")
    else:
        print(f"  Total samples: {args.num_samples}")
    
    selected_indices = select_samples(
        dataset_raw,
        num_samples=args.num_samples,
        classes=args.classes,
        num_per_class=args.num_per_class,
        random_seed=args.seed
    )
    
    print(f"  Selected: {len(selected_indices)} samples")
    
    # Count by class
    class_counts = {}
    for _, label in selected_indices:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("\n  Distribution:")
    for cls in sorted(class_counts.keys()):
        print(f"    Class {cls}: {class_counts[cls]} samples")
    
    # Export samples
    print("\n" + "=" * 70)
    print("Exporting Samples")
    print("=" * 70)
    
    samples_metadata = []
    
    for i, (dataset_idx, label) in enumerate(selected_indices):
        # Get raw and normalized versions
        raw_img, _ = dataset_raw[dataset_idx]
        norm_tensor, _ = dataset_normalized[dataset_idx]
        
        # Export
        metadata = export_sample(
            raw_img, norm_tensor, label, i,
            output_dir, args.formats
        )
        samples_metadata.append(metadata)
        
        if (i + 1) % 10 == 0 or i == len(selected_indices) - 1:
            print(f"  Exported {i + 1}/{len(selected_indices)} samples...")
    
    print("\n[SUCCESS] All samples exported successfully!")
    
    # Save metadata
    print("\n" + "=" * 70)
    print("Saving Metadata")
    print("=" * 70)
    
    metadata_all = {
        'total_samples': len(samples_metadata),
        'formats': args.formats,
        'shape': [1, 28, 28],
        'dtype': 'float32',
        'normalization': {
            'mean': 0.1307,
            'std': 0.3081
        },
        'class_distribution': class_counts,
        'samples': samples_metadata
    }
    
    metadata_path = output_dir / 'samples_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_all, f, indent=2, ensure_ascii=False)
    
    print(f"  Metadata: {metadata_path}")
    
    # Generate C++ loader
    if 'binary' in args.formats:
        print("\nGenerating C++ loader template...")
        cpp_path = create_cpp_loader_template(samples_metadata, output_dir)
        print(f"  C++ header: {cpp_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Export Summary")
    print("=" * 70)
    
    print(f"\nTotal samples exported: {len(samples_metadata)}")
    print(f"Output directory: {output_dir.absolute()}")
    
    if 'binary' in args.formats:
        bin_dir = output_dir / 'binary'
        total_size = sum(f.stat().st_size for f in bin_dir.glob('*.bin')) / 1024
        print(f"\nBinary files:")
        print(f"  Directory: {bin_dir}")
        print(f"  Count: {len(list(bin_dir.glob('*.bin')))}")
        print(f"  Total size: {total_size:.2f} KB")
    
    if 'png' in args.formats:
        img_dir = output_dir / 'images'
        print(f"\nPNG images:")
        print(f"  Directory: {img_dir}")
        print(f"  Count: {len(list(img_dir.glob('*.png')))}")
    
    print("\nMetadata:")
    print(f"  JSON: {metadata_path}")
    if 'binary' in args.formats:
        print(f"  C++ header: {output_dir / 'mnist_loader.h'}")
    
    print("\nNext steps:")
    print("  1. Include mnist_loader.h in your C++ project")
    print("  2. Load samples with mnist_loader::load_sample()")
    print("  3. Run inference and compare with ground truth labels")
    
    print("=" * 70)
    print("[SUCCESS] Export completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
