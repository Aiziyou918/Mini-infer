#!/bin/bash
# Shell script: CUDA GPU inference test for LeNet-5
# Usage:
#   ./test_cuda_inference.sh [BUILD_DIR] [MODEL_FILE] [SAMPLES_DIR] [NUM_SAMPLES] [--compare-cpu]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${1:-$SCRIPT_DIR/../../../build}"
MODEL_FILE="${2:-$SCRIPT_DIR/models/lenet5.onnx}"
SAMPLES_DIR="${3:-$SCRIPT_DIR/test_samples}"
NUM_SAMPLES="${4:-50}"
COMPARE_CPU=""

# Check for --compare-cpu flag
for arg in "$@"; do
    if [ "$arg" = "--compare-cpu" ]; then
        COMPARE_CPU="--compare-cpu"
    fi
done

# Locate executable
EXE_PATH="$BUILD_DIR/bin/lenet5_cuda_inference"
if [ ! -f "$EXE_PATH" ]; then
    EXE_PATH="$BUILD_DIR/Release/bin/lenet5_cuda_inference"
fi
if [ ! -f "$EXE_PATH" ]; then
    EXE_PATH="$BUILD_DIR/Debug/bin/lenet5_cuda_inference"
fi

if [ ! -f "$EXE_PATH" ]; then
    echo "Error: lenet5_cuda_inference not found."
    echo "Checked:"
    echo "  $BUILD_DIR/bin/lenet5_cuda_inference"
    echo "  $BUILD_DIR/Release/bin/lenet5_cuda_inference"
    echo "  $BUILD_DIR/Debug/bin/lenet5_cuda_inference"
    echo ""
    echo "Make sure you built with CUDA support enabled:"
    echo "  cmake -DMINI_INFER_ENABLE_CUDA=ON ..."
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

SAMPLES_BIN="$SAMPLES_DIR/binary"
if [ ! -d "$SAMPLES_BIN" ]; then
    SAMPLES_BIN="$SAMPLES_DIR"
fi

echo "======================================================================"
echo "LeNet-5 CUDA GPU Inference Test Script (Shell)"
echo "======================================================================"
echo ""
echo "Working directory: $(pwd)"
echo ""

echo "Step 1: Running CUDA GPU Inference"
echo "----------------------------------------------------------------------"
echo "Executable : $EXE_PATH"
echo "Model      : $MODEL_FILE"
echo "Samples    : $SAMPLES_BIN"
echo "NumSamples : $NUM_SAMPLES"
echo ""

# Run inference and capture output
OUTPUT_FILE=$(mktemp)
set +e
"$EXE_PATH" "$MODEL_FILE" "$SAMPLES_BIN" "$NUM_SAMPLES" $COMPARE_CPU 2>&1 | tee "$OUTPUT_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[FAIL] CUDA inference failed with exit code $EXIT_CODE."
    rm -f "$OUTPUT_FILE"
    exit $EXIT_CODE
fi

# Parse results
ACC=$(grep -oP "Accuracy:\s+\K[\d\.]+" "$OUTPUT_FILE" || echo "")
TOTAL=$(grep -oP "Total samples:\s+\K\d+" "$OUTPUT_FILE" || echo "")
CORRECT=$(grep -oP "Correct predictions:\s+\K\d+" "$OUTPUT_FILE" || echo "")
AVG_TIME=$(grep -oP "Average time per sample:\s+\K[\d\.]+" "$OUTPUT_FILE" || echo "")
THROUGHPUT=$(grep -oP "Throughput:\s+\K[\d\.]+" "$OUTPUT_FILE" || echo "")

PASS=0
if [ -n "$ACC" ]; then
    ACC_INT=${ACC%.*}
    if [ "$ACC_INT" -ge 99 ] 2>/dev/null; then
        PASS=1
    fi
fi
if [ "$PASS" -eq 0 ] && [ -n "$TOTAL" ] && [ -n "$CORRECT" ]; then
    if [ "$TOTAL" = "$CORRECT" ]; then
        PASS=1
    fi
fi

echo ""
echo "======================================================================"
if [ $PASS -eq 1 ]; then
    echo "[SUCCESS] CUDA GPU inference passed!"
    [ -n "$ACC" ] && echo "  Accuracy: ${ACC}%"
    [ -n "$TOTAL" ] && echo "  Total samples: $TOTAL"
    [ -n "$AVG_TIME" ] && echo "  Avg time/sample: ${AVG_TIME} ms"
    [ -n "$THROUGHPUT" ] && echo "  Throughput: ${THROUGHPUT} samples/sec"
else
    echo "[WARN] Inference succeeded but accuracy check failed or missing metrics."
    [ -n "$ACC" ] && echo "  Accuracy: ${ACC}%"
    [ -n "$TOTAL" ] && echo "  Total samples: $TOTAL"
fi
echo "======================================================================"

rm -f "$OUTPUT_FILE"

if [ $PASS -ne 1 ]; then
    exit 1
fi
