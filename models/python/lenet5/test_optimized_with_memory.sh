#!/bin/bash
# ============================================================================
# LeNet-5 Optimized Inference Test with Memory Planning (Bash)
# This script tests the optimized inference and compares with PyTorch reference
# ============================================================================

BUILD_TYPE=${1:-Debug}

echo "========================================"
echo "LeNet-5 Optimized Inference Test"
echo "Build Type: $BUILD_TYPE"
echo "========================================"
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$SCRIPT_DIR/models/lenet5.onnx"
SAMPLES_DIR="$SCRIPT_DIR/test_samples"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Find executable
EXECUTABLE=""
for candidate in \
    "$SCRIPT_DIR/../../../build/Debug/bin/lenet5_optimized_with_memory_planning" \
    "$SCRIPT_DIR/../../../build/Release/bin/lenet5_optimized_with_memory_planning" \
    "$SCRIPT_DIR/../../../build/bin/lenet5_optimized_with_memory_planning"
do
    if [ -x "$candidate" ]; then
        EXECUTABLE="$candidate"
        break
    fi
done

# Check executable
if [ -z "$EXECUTABLE" ]; then
    echo -e "${RED}[ERROR] Executable not found${NC}"
    echo "Checked:"
    echo "  $SCRIPT_DIR/../../../build/Debug/bin/lenet5_optimized_with_memory_planning"
    echo "  $SCRIPT_DIR/../../../build/Release/bin/lenet5_optimized_with_memory_planning"
    echo "  $SCRIPT_DIR/../../../build/bin/lenet5_optimized_with_memory_planning"
    echo ""
    echo "Please build the project first:"
    echo "  cmake --build build/Debug --parallel"
    exit 1
fi

echo "Using executable: $EXECUTABLE"
echo ""

# Check model
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}[ERROR] Model not found: $MODEL_PATH${NC}"
    exit 1
fi

# Check samples
if [ ! -d "$SAMPLES_DIR" ]; then
    echo -e "${RED}[ERROR] Samples directory not found: $SAMPLES_DIR${NC}"
    exit 1
fi

# Use binary samples if available
SAMPLES_BIN="$SAMPLES_DIR/binary"
if [ -d "$SAMPLES_BIN" ]; then
    SAMPLES_DIR="$SAMPLES_BIN"
fi

echo "========================================"
echo "Step 1: Generate PyTorch Reference"
echo "========================================"
echo ""

python generate_reference_outputs.py \
    --checkpoint checkpoints/lenet5_best.pth \
    --samples-dir test_samples \
    --output test_samples/reference_outputs.json

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Failed to generate reference outputs${NC}"
    exit 1
fi
echo ""

echo "========================================"
echo "Step 2: Run Optimized Inference (WITH Memory Planning)"
echo "========================================"
echo ""

OUTPUT_WITH_MEM="$SCRIPT_DIR/test_samples/optimized_memory_outputs.json"
"$EXECUTABLE" --model "$MODEL_PATH" --samples "$SAMPLES_DIR" --save-outputs "$OUTPUT_WITH_MEM"

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Optimized inference failed${NC}"
    exit 1
fi
echo ""

echo "========================================"
echo "Step 3: Run Optimized Inference (WITHOUT Memory Planning)"
echo "========================================"
echo ""

OUTPUT_NO_MEM="$SCRIPT_DIR/test_samples/optimized_no_memory_outputs.json"
"$EXECUTABLE" --model "$MODEL_PATH" --samples "$SAMPLES_DIR" --no-memory-planning --save-outputs "$OUTPUT_NO_MEM"

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Optimized inference failed${NC}"
    exit 1
fi
echo ""

echo "========================================"
echo "Step 4: Compare with PyTorch Reference"
echo "========================================"
echo ""

python compare_outputs.py \
    --reference test_samples/reference_outputs.json \
    --minfer "$OUTPUT_WITH_MEM" \
    --output test_samples/optimized_comparison_report.json

COMPARE_RESULT=$?
echo ""

echo "========================================"
echo "Step 5: Memory Usage Comparison"
echo "========================================"
echo ""

python compare_memory_usage.py \
    "$OUTPUT_NO_MEM" \
    "$OUTPUT_WITH_MEM"

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo ""

if [ $COMPARE_RESULT -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] Optimized inference matches PyTorch reference!${NC}"
    echo ""
    echo -e "${GREEN}[PASS] All tests passed!${NC}"
else
    echo -e "${RED}[FAILED] Optimized inference does NOT match PyTorch reference${NC}"
    echo ""
    echo -e "${RED}[FAIL] Tests failed${NC}"
fi

echo "========================================"

exit $COMPARE_RESULT
