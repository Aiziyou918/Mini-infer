#!/bin/bash
# Complete End-to-End Test for LeNet-5
# This script should be run from models/python/lenet5 directory
#
# Usage: ./test_lenet5.sh
#
# Steps:
# 1. Generates reference outputs from PyTorch
# 2. Runs C++ inference with Mini-Infer
# 3. Compares the outputs

set -e  # Exit on error

echo "======================================================================"
echo "LeNet-5 End-to-End Test Script"
echo "======================================================================"
echo ""
echo "Working directory: $(pwd)"
echo ""

# Check if we are in the correct directory
if [ ! -f "lenet5_model.py" ]; then
    echo "Error: Script must be run from models/python/lenet5 directory"
    echo ""
    echo "Please run:"
    echo "  cd models/python/lenet5"
    echo "  ./test_lenet5.sh"
    echo ""
    exit 1
fi

# Check if model checkpoint exists
if [ ! -f "checkpoints/lenet5_best.pth" ]; then
    echo "Error: Model checkpoint not found: checkpoints/lenet5_best.pth"
    echo ""
    echo "Please train the model first:"
    echo "  python train_lenet5.py --epochs 10"
    echo ""
    exit 1
fi

# Check if test samples exist
if [ ! -d "test_samples/binary" ]; then
    echo "Error: Test samples not found"
    echo ""
    echo "Please export test samples first:"
    echo "  python export_mnist_samples.py --num-per-class 10"
    echo ""
    exit 1
fi

echo "Step 1: Generating PyTorch Reference Outputs"
echo "----------------------------------------------------------------------"
python3 generate_reference_outputs.py \
    --checkpoint checkpoints/lenet5_best.pth \
    --samples-dir test_samples \
    --output test_samples/reference_outputs.json
echo ""

echo "Step 2: Running C++ Mini-Infer Inference"
echo "----------------------------------------------------------------------"
../../../build/examples/lenet5_inference \
    weights \
    test_samples/binary \
    --save-outputs test_samples/minfer_outputs.json || {
    echo ""
    echo "Note: Make sure lenet5_inference is compiled:"
    echo "  cmake --build build --config Release --target lenet5_inference"
    echo ""
    exit 1
}
echo ""

echo "Step 3: Comparing Outputs"
echo "----------------------------------------------------------------------"
python3 compare_outputs.py \
    --reference test_samples/reference_outputs.json \
    --minfer test_samples/minfer_outputs.json \
    --output test_samples/comparison_report.json

COMPARE_RESULT=$?

echo ""
echo "======================================================================"
if [ $COMPARE_RESULT -eq 0 ]; then
    echo "[SUCCESS] TEST PASSED: Mini-Infer matches PyTorch!"
else
    echo "[FAILED] TEST FAILED: Differences detected"
fi
echo "======================================================================"
echo ""
echo "Generated files:"
echo "  - Reference outputs: test_samples/reference_outputs.json"
echo "  - Mini-Infer outputs: test_samples/minfer_outputs.json"
echo "  - Comparison report: test_samples/comparison_report.json"
echo ""
echo "View detailed report:"
echo "  cat test_samples/comparison_report.json | python3 -m json.tool"
echo ""

exit $COMPARE_RESULT
