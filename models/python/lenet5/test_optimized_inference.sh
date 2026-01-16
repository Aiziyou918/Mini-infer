#!/bin/bash
# Shell script: correctness check for optimized LeNet-5 inference
# Usage:
#   ./test_optimized_inference.sh [BUILD_DIR] [MODEL_FILE] [SAMPLES_DIR] [NUM_SAMPLES]

set -euo pipefail

script_dir="$(cd -- "$(dirname "$0")" && pwd)"
BUILD_DIR="${1:-${script_dir}/../../../build/Debug}"
MODEL_FILE="${2:-${script_dir}/models/lenet5.onnx}"
SAMPLES_DIR="${3:-${script_dir}/test_samples/binary}"
NUM_SAMPLES="${4:-100}"

DEBUG_EXE="${BUILD_DIR}/bin/lenet5_optimized_inference"
RELEASE_EXE="${script_dir}/../../../build/Release/bin/lenet5_optimized_inference"
EXE_PATH="$DEBUG_EXE"
[ -f "$EXE_PATH" ] || EXE_PATH="$RELEASE_EXE"

if [ ! -f "$EXE_PATH" ]; then
    echo "Error: lenet5_optimized_inference not found."
    echo "Checked:"
    echo "  $DEBUG_EXE"
    echo "  $RELEASE_EXE"
    exit 1
fi
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    exit 1
fi
if [ ! -d "$SAMPLES_DIR" ]; then
    echo "Error: Samples directory not found: $SAMPLES_DIR"
    exit 1
fi

echo "======================================================================"
echo "LeNet-5 Optimized End-to-End Test Script (Bash)"
echo "======================================================================"
echo ""
echo "Working directory: $(pwd)"
echo "Build type: $(basename "$(dirname "$(dirname "$EXE_PATH")")")"
echo ""

echo "Step 1: Generating PyTorch Reference Outputs"
echo "----------------------------------------------------------------------"
python generate_reference_outputs.py \
  --checkpoint checkpoints/lenet5_best.pth \
  --samples-dir "$(dirname "$SAMPLES_DIR")" \
  --output "$(dirname "$SAMPLES_DIR")/reference_outputs.json"
if [ $? -ne 0 ]; then
  echo "Error: Failed to generate reference outputs"
  exit 1
fi
echo ""

echo "Step 2: Running Optimized C++ Inference"
echo "----------------------------------------------------------------------"
echo "Executable : $EXE_PATH"
echo "Model      : $MODEL_FILE"
echo "Samples    : $SAMPLES_DIR"
echo "NumSamples : $NUM_SAMPLES"
echo ""

OUT_JSON="$(dirname "$SAMPLES_DIR")/minfer_optimized_outputs.json"
output=$("$EXE_PATH" "$MODEL_FILE" "$SAMPLES_DIR" "$NUM_SAMPLES" --save-outputs "$OUT_JSON" 2>&1)
code=$?
echo "$output"

if [ $code -ne 0 ]; then
  echo
  echo "[FAIL] Inference failed with exit code $code."
  exit $code
fi

acc=$(echo "$output" | sed -n 's/.*Accuracy:[[:space:]]*\([0-9.]\+\)%.*/\1/p' | head -n1)
total=$(echo "$output" | sed -n 's/.*Total samples:[[:space:]]*\([0-9]\+\).*/\1/p' | head -n1)
corr=$(echo "$output" | sed -n 's/.*Correct predictions:[[:space:]]*\([0-9]\+\).*/\1/p' | head -n1)

pass=0
if [ -n "$acc" ] && [ -n "$total" ]; then
  if awk "BEGIN {exit !($acc >= 99.0)}"; then
    pass=1
  elif [ -n "$corr" ] && [ "$corr" = "$total" ]; then
    pass=1
  fi
fi

if [ $pass -eq 1 ]; then
  echo
  echo "[SUCCESS] Optimized inference passed accuracy check. Accuracy=${acc:-N/A}%, Total=${total:-N/A}"
else
  echo
  echo "[WARN] Inference succeeded but accuracy check failed or missing metrics. Accuracy=${acc:-N/A}%, Total=${total:-N/A}"
fi

echo ""
echo "Step 3: Comparing Outputs"
echo "----------------------------------------------------------------------"
python compare_outputs.py \
  --reference "$(dirname "$SAMPLES_DIR")/reference_outputs.json" \
  --minfer "$OUT_JSON" \
  --output "$(dirname "$SAMPLES_DIR")/comparison_optimized.json"
compare_code=$?

echo ""
echo "======================================================================"
if [ $compare_code -eq 0 ] && [ $pass -eq 1 ]; then
  echo "[SUCCESS] TEST PASSED: Optimized Mini-Infer matches PyTorch!"
elif [ $compare_code -eq 0 ]; then
  echo "[WARN] Outputs match but runtime accuracy threshold not met."
else
  echo "[FAILED] TEST FAILED: Differences detected"
fi
echo "======================================================================"
echo ""
echo "Generated files:"
echo "  - Reference outputs: $(dirname "$SAMPLES_DIR")/reference_outputs.json"
echo "  - Mini-Infer outputs: $OUT_JSON"
echo "  - Comparison report: $(dirname "$SAMPLES_DIR")/comparison_optimized.json"

if [ $compare_code -ne 0 ] || [ $pass -ne 1 ]; then
  exit 1
fi

echo ""
echo "Note: Optimized binary does not emit per-sample outputs; logits comparison is unavailable here."

