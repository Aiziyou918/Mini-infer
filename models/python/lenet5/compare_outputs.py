#!/usr/bin/env python3
"""
Compare outputs between PyTorch reference and C++ Mini-Infer implementation
"""

import json
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple


class OutputComparator:
    """Compare inference outputs between PyTorch and Mini-Infer"""
    
    def __init__(self, reference_file: str, minfer_file: str):
        """
        Args:
            reference_file: JSON file with PyTorch reference outputs
            minfer_file: JSON file with Mini-Infer outputs
        """
        with open(reference_file, 'r', encoding='utf-8') as f:
            self.reference = json.load(f)
        
        with open(minfer_file, 'r', encoding='utf-8') as f:
            self.minfer = json.load(f)
        
        # Create lookup by filename
        self.ref_by_file = {
            r['filename']: r for r in self.reference['results']
        }
        self.minfer_by_file = {
            m['filename']: m for m in self.minfer['results']
        }
    
    def compute_metrics(self) -> Dict:
        """Compute comparison metrics"""
        
        # Match samples
        common_files = set(self.ref_by_file.keys()) & set(self.minfer_by_file.keys())
        
        if not common_files:
            raise ValueError("No common samples found between reference and Mini-Infer outputs")
        
        # Statistics
        total_samples = len(common_files)
        prediction_match = 0
        logits_errors = []
        prob_errors = []
        max_logit_error = 0
        max_prob_error = 0
        
        mismatches = []
        
        for filename in sorted(common_files):
            ref = self.ref_by_file[filename]
            minfer = self.minfer_by_file[filename]
            
            # Check prediction match
            if ref['predicted'] == minfer['predicted']:
                prediction_match += 1
            else:
                mismatches.append({
                    'filename': filename,
                    'label': ref['label'],
                    'pytorch_pred': ref['predicted'],
                    'minfer_pred': minfer['predicted'],
                    'pytorch_conf': ref['confidence'],
                    'minfer_conf': minfer['confidence']
                })
            
            # Compute logits error
            ref_logits = np.array(ref['logits'])
            minfer_logits = np.array(minfer['logits'])
            logit_error = np.abs(ref_logits - minfer_logits)
            logits_errors.append(logit_error)
            max_logit_error = max(max_logit_error, logit_error.max())
            
            # Compute probability error
            ref_probs = np.array(ref['probabilities'])
            minfer_probs = np.array(minfer['probabilities'])
            prob_error = np.abs(ref_probs - minfer_probs)
            prob_errors.append(prob_error)
            max_prob_error = max(max_prob_error, prob_error.max())
        
        # Aggregate statistics
        logits_errors = np.array(logits_errors)
        prob_errors = np.array(prob_errors)
        
        metrics = {
            'total_samples': total_samples,
            'prediction_accuracy': 100.0 * prediction_match / total_samples,
            'prediction_match_count': prediction_match,
            'logits': {
                'mean_absolute_error': float(logits_errors.mean()),
                'max_absolute_error': float(max_logit_error),
                'std_error': float(logits_errors.std()),
                'median_error': float(np.median(logits_errors))
            },
            'probabilities': {
                'mean_absolute_error': float(prob_errors.mean()),
                'max_absolute_error': float(max_prob_error),
                'std_error': float(prob_errors.std()),
                'median_error': float(np.median(prob_errors))
            },
            'mismatches': mismatches
        }
        
        return metrics
    
    def print_report(self, metrics: Dict):
        """Print comparison report"""
        
        print("=" * 70)
        print("PyTorch vs Mini-Infer Comparison Report")
        print("=" * 70)
        print()
        
        print("Sample Statistics:")
        print(f"  Total samples compared: {metrics['total_samples']}")
        print()
        
        print("Prediction Accuracy:")
        acc = metrics['prediction_accuracy']
        match_count = metrics['prediction_match_count']
        total = metrics['total_samples']
        print(f"  Matches: {match_count}/{total} ({acc:.2f}%)")
        
        if acc == 100.0:
            print("  [SUCCESS] Perfect match!")
        else:
            print(f"  [FAILED] {total - match_count} mismatches")
        print()
        
        print("Logits Comparison:")
        logits = metrics['logits']
        print(f"  Mean Absolute Error: {logits['mean_absolute_error']:.6f}")
        print(f"  Max Absolute Error:  {logits['max_absolute_error']:.6f}")
        print(f"  Median Error:        {logits['median_error']:.6f}")
        print(f"  Std Error:           {logits['std_error']:.6f}")
        
        if logits['max_absolute_error'] < 1e-4:
            print("  [SUCCESS] Excellent agreement (< 1e-4)")
        elif logits['max_absolute_error'] < 1e-3:
            print("  [SUCCESS] Good agreement (< 1e-3)")
        elif logits['max_absolute_error'] < 1e-2:
            print("  ⚠ Acceptable agreement (< 1e-2)")
        else:
            print("  [FAILED] Poor agreement (>= 1e-2)")
        print()
        
        print("Probabilities Comparison:")
        probs = metrics['probabilities']
        print(f"  Mean Absolute Error: {probs['mean_absolute_error']:.6f}")
        print(f"  Max Absolute Error:  {probs['max_absolute_error']:.6f}")
        print(f"  Median Error:        {probs['median_error']:.6f}")
        print(f"  Std Error:           {probs['std_error']:.6f}")
        
        if probs['max_absolute_error'] < 1e-5:
            print("  [SUCCESS] Excellent agreement (< 1e-5)")
        elif probs['max_absolute_error'] < 1e-4:
            print("  [SUCCESS] Good agreement (< 1e-4)")
        elif probs['max_absolute_error'] < 1e-3:
            print("  ⚠ Acceptable agreement (< 1e-3)")
        else:
            print("  [FAILED] Poor agreement (>= 1e-3)")
        print()
        
        # Show mismatches if any
        if metrics['mismatches']:
            print("Prediction Mismatches:")
            for mm in metrics['mismatches']:
                print(f"  {mm['filename']}:")
                print(f"    Label: {mm['label']}")
                print(f"    PyTorch: {mm['pytorch_pred']} (conf={mm['pytorch_conf']:.4f})")
                print(f"    Mini-Infer: {mm['minfer_pred']} (conf={mm['minfer_conf']:.4f})")
            print()
        
        print("=" * 70)
        
        # Overall assessment
        if acc == 100.0 and logits['max_absolute_error'] < 1e-3:
            print("[SUCCESS] PASS: Mini-Infer implementation matches PyTorch!")
            return True
        else:
            print("[FAILED] FAIL: Significant differences detected")
            return False
    
    def save_detailed_report(self, metrics: Dict, output_file: str):
        """Save detailed comparison report to JSON"""
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nDetailed report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare PyTorch and Mini-Infer outputs'
    )
    parser.add_argument(
        '--reference',
        type=str,
        default='./test_samples/reference_outputs.json',
        help='PyTorch reference outputs JSON file'
    )
    parser.add_argument(
        '--minfer',
        type=str,
        default='./test_samples/minfer_outputs.json',
        help='Mini-Infer outputs JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./test_samples/comparison_report.json',
        help='Output comparison report JSON file'
    )
    
    args = parser.parse_args()
    
    try:
        comparator = OutputComparator(args.reference, args.minfer)
        metrics = comparator.compute_metrics()
        passed = comparator.print_report(metrics)
        comparator.save_detailed_report(metrics, args.output)
        
        # Exit with appropriate code
        exit(0 if passed else 1)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("  1. Run: python generate_reference_outputs.py")
        print("  2. Run: lenet5_inference.exe (with --save-outputs)")
        print("  3. Then run this comparison script")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
