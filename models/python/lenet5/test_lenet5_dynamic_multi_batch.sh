#!/bin/bash
# Bash script for exercising lenet5_dynamic_multi_batch example
# Usage:
#   ./test_lenet5_dynamic_multi_batch.sh [--max-batch N] [--accuracy-threshold X]
#                                        [--model path] [--samples path] [--exe path]

set -euo pipefail

MAX_BATCH=16
ACCURACY_THRESHOLD=0.9
MODEL_PATH=""
SAMPLES_DIR=""
EXECUTABLE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-batch)
            MAX_BATCH="$2"; shift 2 ;;
        --accuracy-threshold)
            ACCURACY_THRESHOLD="$2"; shift 2 ;;
        --model)
            MODEL_PATH="$2"; shift 2 ;;
        --samples)
            SAMPLES_DIR="$2"; shift 2 ;;
        --exe)
            EXECUTABLE="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "LeNet-5 Dynamic Multi-Batch Test (Bash)"
echo "======================================================================"
echo "Working directory : $PWD"
echo "Requested max batch: $MAX_BATCH"
echo ""

if [[ ! -f "lenet5_model.py" ]]; then
    echo "[ERROR] Run this script from models/python/lenet5" >&2
    exit 1
fi

if [[ ! -f "checkpoints/lenet5_best.pth" ]]; then
    echo "[ERROR] Missing checkpoint checkpoints/lenet5_best.pth" >&2
    echo "Train the model first: python3 train_lenet5.py --epochs 10"
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-$SCRIPT_DIR/models/lenet5.onnx}"
SAMPLES_DIR="${SAMPLES_DIR:-$SCRIPT_DIR/test_samples/dynamic_multi_batch}"
BINARY_DIR="$SAMPLES_DIR/binary"
CHECKPOINT="$SCRIPT_DIR/checkpoints/lenet5_best.pth"

# Build desired batch sizes identical to the C++ demo
declare -a RAW_BATCHES=(1 4 8 12)
if (( MAX_BATCH > 16 )); then
    RAW_BATCHES+=("16")
else
    RAW_BATCHES+=("$MAX_BATCH")
fi
if (( MAX_BATCH <= 0 )); then
    RAW_BATCHES=("$MAX_BATCH")
fi

readarray -t BATCHES < <(printf "%s\n" "${RAW_BATCHES[@]}" | awk -v max="$MAX_BATCH" '$1>0 && $1<=max {print $1}' | sort -n -u)
if [[ "${#BATCHES[@]}" -eq 0 ]]; then
    BATCHES=("$MAX_BATCH")
fi
SAMPLES_REQUIRED=0
for val in "${BATCHES[@]}"; do
    SAMPLES_REQUIRED=$((SAMPLES_REQUIRED + val))
done

echo "Target batch sizes  : ${BATCHES[*]}"
echo "Required sample count: $SAMPLES_REQUIRED"
echo ""

# Ensure ONNX
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Step 1: Exporting ONNX model"
    echo "----------------------------------------------------------------------"
    python3 export_lenet5.py \
        --checkpoint "$CHECKPOINT" \
        --format onnx \
        --output "$MODEL_PATH"
    echo ""
fi

count_labeled_samples() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo "0"
        return
    fi
    find "$dir" -maxdepth 1 -type f -name "*_label_*.bin" | wc -l | awk '{print $1}'
}

LABEL_COUNT=$(count_labeled_samples "$BINARY_DIR")
if (( LABEL_COUNT < SAMPLES_REQUIRED )); then
    NUM_PER_CLASS=$(( (SAMPLES_REQUIRED + 9) / 10 ))
    if (( NUM_PER_CLASS < 5 )); then NUM_PER_CLASS=5; fi

    echo "Step 2: Exporting MNIST samples (num-per-class=$NUM_PER_CLASS)"
    echo "----------------------------------------------------------------------"
    python3 export_mnist_samples.py \
        --output-dir "$SAMPLES_DIR" \
        --num-per-class "$NUM_PER_CLASS" \
        --formats binary png
    echo ""
    LABEL_COUNT=$(count_labeled_samples "$BINARY_DIR")
fi

if (( LABEL_COUNT < SAMPLES_REQUIRED )); then
    echo "[ERROR] Only $LABEL_COUNT labeled samples available under $BINARY_DIR" >&2
    exit 1
fi

echo "Sample directory    : $BINARY_DIR"
echo "Available samples   : $LABEL_COUNT"
echo ""

if [[ -z "$EXECUTABLE" ]]; then
    for candidate in \
        "$SCRIPT_DIR/../../../build/examples/lenet5_dynamic_multi_batch" \
        "$SCRIPT_DIR/../../../build/Debug/bin/lenet5_dynamic_multi_batch" \
        "$SCRIPT_DIR/../../../build/Release/bin/lenet5_dynamic_multi_batch"
    do
        if [[ -x "$candidate" ]]; then
            EXECUTABLE="$candidate"
            break
        fi
    done
fi

if [[ -z "$EXECUTABLE" ]]; then
    echo "[ERROR] lenet5_dynamic_multi_batch executable not found. Build it first." >&2
    echo "Hint: cmake --build build --config Release --target lenet5_dynamic_multi_batch"
    exit 1
fi

echo "Using executable    : $EXECUTABLE"
echo "Accuracy threshold  : $ACCURACY_THRESHOLD"
echo ""

echo "Step 3: Running lenet5_dynamic_multi_batch"
echo "----------------------------------------------------------------------"
set +e
"$EXECUTABLE" \
    --model "$MODEL_PATH" \
    --samples "$BINARY_DIR" \
    --accuracy-threshold "$ACCURACY_THRESHOLD" \
    --max-batch "$MAX_BATCH"
EXIT_CODE=$?
set -e
echo ""

echo "======================================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[SUCCESS] Dynamic multi-batch test completed."
else
    echo "[FAILED] Dynamic multi-batch test failed with exit code $EXIT_CODE."
fi
echo "======================================================================"
echo "Model path   : $MODEL_PATH"
echo "Samples used : $BINARY_DIR"
echo "Batch sizes  : ${BATCHES[*]}"
echo ""

exit $EXIT_CODE
