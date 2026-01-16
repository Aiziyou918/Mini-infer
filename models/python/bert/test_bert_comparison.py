#!/usr/bin/env python3
"""
BERT Inference Comparison Test

对比 Mini-Infer 和 PyTorch 的 BERT 推理结果

Usage:
    python test_bert_comparison.py
    python test_bert_comparison.py --mini-infer-bin ../../build/Debug/bin/bert_inference
"""

import argparse
import json
import subprocess
import sys
import re
from pathlib import Path
from typing import Optional
import numpy as np


def parse_mini_infer_output(output: str) -> dict:
    """解析 Mini-Infer 的输出"""
    results = []

    # 匹配样本结果行
    # Sample    0: pred=1 (positive), conf=0.5948, label=1 (positive) [CORRECT]
    pattern = r'Sample\s+(\d+):\s+pred=(\d+)\s+\((\w+)\),\s+conf=([\d.]+),\s+label=(\d+)\s+\((\w+)\)\s+\[(\w+)\]'

    for match in re.finditer(pattern, output):
        idx = int(match.group(1))
        pred = int(match.group(2))
        pred_name = match.group(3)
        conf = float(match.group(4))
        label = int(match.group(5))
        label_name = match.group(6)
        status = match.group(7)

        results.append({
            "index": idx,
            "predicted": pred,
            "predicted_name": pred_name,
            "confidence": conf,
            "label": label,
            "label_name": label_name,
            "correct": status == "CORRECT"
        })

    # 解析汇总信息
    accuracy_match = re.search(r'Accuracy:\s+([\d.]+)%', output)
    total_match = re.search(r'Total samples:\s+(\d+)', output)
    correct_match = re.search(r'Correct:\s+(\d+)\s+/', output)
    time_match = re.search(r'Average time per sample:\s+([\d.]+)\s+ms', output)

    return {
        "results": results,
        "total_samples": int(total_match.group(1)) if total_match else len(results),
        "correct": int(correct_match.group(1)) if correct_match else sum(1 for r in results if r["correct"]),
        "accuracy": float(accuracy_match.group(1)) if accuracy_match else 0.0,
        "avg_time_ms": float(time_match.group(1)) if time_match else 0.0
    }


def run_mini_infer(
    mini_infer_bin: str,
    model_path: str,
    samples_dir: str,
    num_samples: int
) -> Optional[dict]:
    """运行 Mini-Infer 推理"""
    cmd = [mini_infer_bin, model_path, samples_dir, str(num_samples)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"[ERROR] Mini-Infer failed with return code {result.returncode}")
            print(f"stderr: {result.stderr}")
            return None

        return parse_mini_infer_output(result.stdout)

    except subprocess.TimeoutExpired:
        print("[ERROR] Mini-Infer timed out")
        return None
    except FileNotFoundError:
        print(f"[ERROR] Mini-Infer binary not found: {mini_infer_bin}")
        return None


def load_pytorch_reference(samples_dir: str) -> Optional[dict]:
    """加载 PyTorch 参考输出"""
    ref_path = Path(samples_dir) / "reference_outputs.json"

    if not ref_path.exists():
        print(f"[ERROR] Reference file not found: {ref_path}")
        return None

    with open(ref_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_results(mini_infer: dict, pytorch: dict, tolerance: float = 0.01) -> dict:
    """对比 Mini-Infer 和 PyTorch 的结果"""
    comparison = {
        "total_samples": mini_infer["total_samples"],
        "mini_infer_accuracy": mini_infer["accuracy"],
        "pytorch_accuracy": pytorch["accuracy"],
        "prediction_match": 0,
        "confidence_diff": [],
        "details": []
    }

    mi_results = {r["index"]: r for r in mini_infer["results"]}
    pt_results = {r["index"]: r for r in pytorch["results"]}

    for idx in sorted(mi_results.keys()):
        if idx not in pt_results:
            continue

        mi = mi_results[idx]
        pt = pt_results[idx]

        pred_match = mi["predicted"] == pt["predicted"]
        if pred_match:
            comparison["prediction_match"] += 1

        # 计算置信度差异 (Mini-Infer confidence vs PyTorch probability)
        pt_conf = pt["probabilities"][pt["predicted"]]
        conf_diff = abs(mi["confidence"] - pt_conf)
        comparison["confidence_diff"].append(conf_diff)

        comparison["details"].append({
            "index": idx,
            "text": pt.get("text", ""),
            "label": mi["label"],
            "mini_infer_pred": mi["predicted"],
            "pytorch_pred": pt["predicted"],
            "mini_infer_conf": mi["confidence"],
            "pytorch_conf": pt_conf,
            "pytorch_logits": pt["logits"],
            "prediction_match": pred_match,
            "confidence_diff": conf_diff
        })

    # 计算统计信息
    if comparison["confidence_diff"]:
        comparison["avg_confidence_diff"] = np.mean(comparison["confidence_diff"])
        comparison["max_confidence_diff"] = np.max(comparison["confidence_diff"])
    else:
        comparison["avg_confidence_diff"] = 0.0
        comparison["max_confidence_diff"] = 0.0

    comparison["prediction_match_rate"] = (
        comparison["prediction_match"] / comparison["total_samples"] * 100
        if comparison["total_samples"] > 0 else 0.0
    )

    return comparison


def print_comparison_report(comparison: dict, verbose: bool = False):
    """打印对比报告"""
    print()
    print("=" * 70)
    print("BERT Inference Comparison Report")
    print("=" * 70)
    print()

    print(f"Total Samples: {comparison['total_samples']}")
    print()

    print("Accuracy Comparison:")
    print(f"  Mini-Infer: {comparison['mini_infer_accuracy']:.2f}%")
    print(f"  PyTorch:    {comparison['pytorch_accuracy']:.2f}%")
    print()

    print("Prediction Consistency:")
    print(f"  Matching predictions: {comparison['prediction_match']} / {comparison['total_samples']}")
    print(f"  Match rate: {comparison['prediction_match_rate']:.2f}%")
    print()

    print("Confidence/Probability Difference:")
    print(f"  Average: {comparison['avg_confidence_diff']:.6f}")
    print(f"  Maximum: {comparison['max_confidence_diff']:.6f}")
    print()

    if verbose:
        print("-" * 70)
        print("Detailed Results:")
        print("-" * 70)
        for detail in comparison["details"]:
            status = "MATCH" if detail["prediction_match"] else "DIFF"
            print(f"\nSample {detail['index']:04d}: [{status}]")
            print(f"  Text: {detail['text'][:60]}...")
            print(f"  Label: {detail['label']}")
            print(f"  Mini-Infer: pred={detail['mini_infer_pred']}, conf={detail['mini_infer_conf']:.4f}")
            print(f"  PyTorch:    pred={detail['pytorch_pred']}, conf={detail['pytorch_conf']:.4f}")
            print(f"  PyTorch logits: {detail['pytorch_logits']}")
            print(f"  Confidence diff: {detail['confidence_diff']:.6f}")

    print()
    print("=" * 70)

    # 判断测试是否通过
    passed = comparison["prediction_match_rate"] >= 100.0
    if passed:
        print("[PASS] All predictions match between Mini-Infer and PyTorch!")
    else:
        print(f"[WARN] Prediction mismatch: {comparison['total_samples'] - comparison['prediction_match']} samples differ")

    print("=" * 70)

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Compare BERT inference results between Mini-Infer and PyTorch"
    )
    parser.add_argument(
        "--mini-infer-bin",
        type=str,
        default=None,
        help="Path to Mini-Infer bert_inference binary"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/bert_tiny.onnx",
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="./test_samples",
        help="Directory containing test samples"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed comparison"
    )
    parser.add_argument(
        "--regenerate-reference",
        action="store_true",
        help="Regenerate PyTorch reference outputs before comparison"
    )

    args = parser.parse_args()

    # 确定脚本所在目录
    script_dir = Path(__file__).parent.resolve()

    # 设置默认路径
    if args.mini_infer_bin is None:
        # 尝试多个可能的路径
        possible_paths = [
            script_dir / "../../../build/Debug/bin/bert_inference",
            script_dir / "../../../build/Release/bin/bert_inference",
            script_dir / "../../../build/bin/bert_inference",
        ]
        for p in possible_paths:
            if p.exists():
                args.mini_infer_bin = str(p.resolve())
                break

        if args.mini_infer_bin is None:
            print("[ERROR] Could not find bert_inference binary. Please specify with --mini-infer-bin")
            sys.exit(1)

    # 解析相对路径
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = script_dir / model_path

    samples_dir = Path(args.samples_dir)
    if not samples_dir.is_absolute():
        samples_dir = script_dir / samples_dir

    print("=" * 70)
    print("BERT Inference Comparison Test")
    print("=" * 70)
    print()
    print(f"Mini-Infer binary: {args.mini_infer_bin}")
    print(f"ONNX model: {model_path}")
    print(f"Samples directory: {samples_dir}")
    print(f"Number of samples: {args.num_samples}")
    print()

    # 检查文件是否存在
    if not Path(args.mini_infer_bin).exists():
        print(f"[ERROR] Mini-Infer binary not found: {args.mini_infer_bin}")
        sys.exit(1)

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    if not samples_dir.exists():
        print(f"[ERROR] Samples directory not found: {samples_dir}")
        sys.exit(1)

    # 重新生成参考输出（如果需要）
    if args.regenerate_reference:
        print("Regenerating PyTorch reference outputs...")
        ref_script = script_dir / "generate_reference_outputs.py"
        if ref_script.exists():
            subprocess.run([
                sys.executable, str(ref_script),
                "--samples-dir", str(samples_dir)
            ], check=True)
            print()

    # 加载 PyTorch 参考输出
    print("Loading PyTorch reference outputs...")
    pytorch_results = load_pytorch_reference(str(samples_dir))
    if pytorch_results is None:
        print("[ERROR] Failed to load PyTorch reference outputs")
        print("Run 'python generate_reference_outputs.py' first")
        sys.exit(1)
    print(f"Loaded {pytorch_results['total_samples']} reference samples")
    print()

    # 运行 Mini-Infer
    print("Running Mini-Infer inference...")
    mini_infer_results = run_mini_infer(
        args.mini_infer_bin,
        str(model_path),
        str(samples_dir),
        args.num_samples
    )

    if mini_infer_results is None:
        print("[ERROR] Failed to run Mini-Infer")
        sys.exit(1)

    print(f"Mini-Infer processed {mini_infer_results['total_samples']} samples")
    print(f"Average inference time: {mini_infer_results['avg_time_ms']:.2f} ms/sample")

    # 对比结果
    comparison = compare_results(mini_infer_results, pytorch_results)

    # 打印报告
    passed = print_comparison_report(comparison, args.verbose)

    # 返回退出码
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
