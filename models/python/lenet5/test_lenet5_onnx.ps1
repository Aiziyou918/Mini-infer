# Complete End-to-End Test for LeNet-5 ONNX (PowerShell)
# This script should be run from models\python\lenet5 directory
#
# Usage: .\test_lenet5_onnx.ps1 [-BuildType Debug|Release]
#
# Steps:
# 1. Exports ONNX model (if not exists)
# 2. Generates reference outputs from PyTorch
# 3. Runs C++ inference with Mini-Infer ONNX importer
# 4. Compares the outputs

param(
    [string]$BuildType = "Debug"
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "LeNet-5 ONNX End-to-End Test Script (PowerShell)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Working directory: $PWD"
Write-Host "Build type: $BuildType"
Write-Host ""

# Check if we are in the correct directory
if (-not (Test-Path "lenet5_model.py")) {
    Write-Host "[ERROR] Script must be run from models\python\lenet5 directory" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run:"
    Write-Host "  cd models\python\lenet5"
    Write-Host "  .\test_lenet5_onnx.ps1"
    Write-Host ""
    exit 1
}

# Check if model checkpoint exists
if (-not (Test-Path "checkpoints\lenet5_best.pth")) {
    Write-Host "[ERROR] Model checkpoint not found: checkpoints\lenet5_best.pth" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please train the model first:"
    Write-Host "  python train_lenet5.py --epochs 10"
    Write-Host ""
    exit 1
}

# Check if test samples exist
if (-not (Test-Path "test_samples\binary")) {
    Write-Host "[ERROR] Test samples not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please export test samples first:"
    Write-Host "  python export_mnist_samples.py --num-per-class 10"
    Write-Host ""
    exit 1
}

# Export ONNX model if not exists
if (-not (Test-Path "models\lenet5.onnx")) {
    Write-Host "Step 1: Exporting ONNX Model" -ForegroundColor Yellow
    Write-Host "----------------------------------------------------------------------"
    python export_onnx.py `
        --checkpoint checkpoints\lenet5_best.pth `
        --output models\lenet5.onnx `
        --opset-version 11
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to export ONNX model" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
} else {
    Write-Host "Step 1: ONNX Model Already Exists" -ForegroundColor Yellow
    Write-Host "----------------------------------------------------------------------"
    Write-Host "Using existing ONNX model: models\lenet5.onnx" -ForegroundColor Green
    Write-Host ""
}

Write-Host "Step 2: Generating PyTorch Reference Outputs" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------"
python generate_reference_outputs.py `
    --checkpoint checkpoints\lenet5_best.pth `
    --samples-dir test_samples `
    --output test_samples\reference_outputs.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to generate reference outputs" -ForegroundColor Red
    exit 1
}
Write-Host ""

Write-Host "Step 3: Running C++ Mini-Infer ONNX Inference" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------"
$exePath = "..\..\..\build\$BuildType\bin\lenet5_onnx_test.exe"

if (-not (Test-Path $exePath)) {
    Write-Host ""
    Write-Host "[ERROR] Executable not found: $exePath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Note: Make sure lenet5_onnx_test is compiled:"
    Write-Host "  cmake --build build --config $BuildType --target lenet5_onnx_test"
    Write-Host ""
    exit 1
}

& $exePath `
    models\lenet5.onnx `
    test_samples\binary `
    --save-outputs test_samples\minfer_onnx_outputs.json

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] C++ inference failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

Write-Host "Step 4: Comparing Outputs" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------"
python compare_outputs.py `
    --reference test_samples\reference_outputs.json `
    --minfer test_samples\minfer_onnx_outputs.json `
    --output test_samples\onnx_comparison_report.json

$compareResult = $LASTEXITCODE

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
if ($compareResult -eq 0) {
    Write-Host "[SUCCESS] TEST PASSED: Mini-Infer ONNX matches PyTorch!" -ForegroundColor Green
} else {
    Write-Host "[FAILED] TEST FAILED: Differences detected" -ForegroundColor Red
}
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Generated files:"
Write-Host "  - ONNX model: models\lenet5.onnx"
Write-Host "  - Reference outputs: test_samples\reference_outputs.json"
Write-Host "  - Mini-Infer ONNX outputs: test_samples\minfer_onnx_outputs.json"
Write-Host "  - Comparison report: test_samples\onnx_comparison_report.json"
Write-Host ""
Write-Host "View detailed report:"
Write-Host "  Get-Content test_samples\onnx_comparison_report.json"
Write-Host "  Get-Content test_samples\onnx_comparison_report.json | ConvertFrom-Json | ConvertTo-Json -Depth 10"
Write-Host ""

exit $compareResult
