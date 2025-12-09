# Complete End-to-End Test for LeNet-5
# This script should be run from models/python/lenet5 directory
# 
# Usage: .\test_lenet5.ps1
#
# Steps:
# 1. Generates reference outputs from PyTorch
# 2. Runs C++ inference with Mini-Infer
# 3. Compares the outputs

# Stop on errors
$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "LeNet-5 End-to-End Test Script" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Working directory: $(Get-Location)"
Write-Host ""

# Check if we are in the correct directory
if (-not (Test-Path "lenet5_model.py")) {
    Write-Host "Error: Script must be run from models/python/lenet5 directory" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run:"
    Write-Host "  cd models\python\lenet5"
    Write-Host "  .\test_lenet5.ps1"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if model checkpoint exists
if (-not (Test-Path "checkpoints\lenet5_best.pth")) {
    Write-Host "Error: Model checkpoint not found: checkpoints\lenet5_best.pth" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please train the model first:"
    Write-Host "  python train_lenet5.py --epochs 10"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if test samples exist
if (-not (Test-Path "test_samples\binary")) {
    Write-Host "Error: Test samples not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please export test samples first:"
    Write-Host "  python export_mnist_samples.py --num-per-class 10"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Step 1: Generating PyTorch Reference Outputs" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------"
python generate_reference_outputs.py `
    --checkpoint checkpoints\lenet5_best.pth `
    --samples-dir test_samples `
    --output test_samples\reference_outputs.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to generate reference outputs" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

Write-Host "Step 2: Running C++ Mini-Infer Inference" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------"

$ScriptDir   = Split-Path -Parent $PSCommandPath
$DebugExe    = Join-Path $ScriptDir "..\..\..\build\Debug\bin\lenet5_inference.exe"
$ReleaseExe  = Join-Path $ScriptDir "..\..\..\build\Release\bin\lenet5_inference.exe"

if (Test-Path $DebugExe) {
    $inferenceExe = $DebugExe
} elseif (Test-Path $ReleaseExe) {
    $inferenceExe = $ReleaseExe
} else {
    Write-Host "Error: lenet5_inference.exe not found in Debug or Release." -ForegroundColor Red
    Write-Host "Checked:" -ForegroundColor Yellow
    Write-Host "  $DebugExe"
    Write-Host "  $ReleaseExe"
    Write-Host ""
    Write-Host "Hint: build the target:" -ForegroundColor Yellow
    Write-Host "  cmake --build build --config Debug --target lenet5_inference"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Using executable: $inferenceExe" -ForegroundColor Green

& $inferenceExe `
    weights `
    test_samples\binary `
    --save-outputs test_samples\minfer_outputs.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to run C++ inference" -ForegroundColor Red
    Write-Host ""
    Write-Host "Note: Make sure lenet5_inference is compiled:" -ForegroundColor Yellow
    Write-Host "  cmake --build build --config Debug --target lenet5_inference"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

Write-Host "Step 3: Comparing Outputs" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------"
python compare_outputs.py `
    --reference test_samples\reference_outputs.json `
    --minfer test_samples\minfer_outputs.json `
    --output test_samples\comparison_report.json

$compareResult = $LASTEXITCODE

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
if ($compareResult -eq 0) {
    Write-Host "[SUCCESS] TEST PASSED: Mini-Infer matches PyTorch!" -ForegroundColor Green
} else {
    Write-Host "[FAILED] TEST FAILED: Differences detected" -ForegroundColor Red
}
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Generated files:"
Write-Host "  - Reference outputs: test_samples\reference_outputs.json"
Write-Host "  - Mini-Infer outputs: test_samples\minfer_outputs.json"
Write-Host "  - Comparison report: test_samples\comparison_report.json"
Write-Host ""
Write-Host "View detailed report:"
Write-Host "  Get-Content test_samples\comparison_report.json | ConvertFrom-Json | ConvertTo-Json -Depth 10"
Write-Host ""

Read-Host "Press Enter to exit"
exit $compareResult
