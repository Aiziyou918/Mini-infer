# PowerShell script for exercising lenet5_dynamic_multi_batch example
# Usage:
#   .\test_lenet5_dynamic_multi_batch.ps1 [-MaxBatch 16] [-AccuracyThreshold 0.9]
#                                         [-ModelPath <path>] [-SamplesDir <path>] [-Executable <path>]

param(
    [ValidateRange(1,256)]
    [int]$MaxBatch = 16,
    [double]$AccuracyThreshold = 0.9,
    [string]$ModelPath = "",
    [string]$SamplesDir = "",
    [string]$Executable = ""
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "LeNet-5 Dynamic Multi-Batch Test (PowerShell)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Working directory : $PWD"
Write-Host "Requested max batch: $MaxBatch"
Write-Host ""

if (-not (Test-Path "lenet5_model.py")) {
    Write-Host "[ERROR] Please run this script from models\python\lenet5." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "checkpoints\lenet5_best.pth")) {
    Write-Host "[ERROR] Missing checkpoint: checkpoints\lenet5_best.pth" -ForegroundColor Red
    Write-Host "        Train the model first: python train_lenet5.py --epochs 10"
    exit 1
}

$ScriptDir = Split-Path -Parent $PSCommandPath
if (-not $ModelPath) {
    $ModelPath = Join-Path $ScriptDir "models/lenet5.onnx"
}
if (-not $SamplesDir) {
    $SamplesDir = Join-Path $ScriptDir "test_samples/dynamic_multi_batch"
}

$binaryDir = Join-Path $SamplesDir "binary"
$checkpointPath = Join-Path $ScriptDir "checkpoints/lenet5_best.pth"

function Get-LabeledSampleCount([string]$DirPath) {
    if (-not (Test-Path $DirPath)) { return 0 }
    return (Get-ChildItem $DirPath -Filter "*.bin" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*_label_*" }).Count
}

# Determine batches like the C++ example and total samples required
$batchSizes = New-Object System.Collections.Generic.List[int]
$batchSizes.Add(1); $batchSizes.Add(4); $batchSizes.Add(8); $batchSizes.Add(12)
$batchSizes.Add([Math]::Min(16, [Math]::Max(1, $MaxBatch)))
$batchSizes = $batchSizes | Sort-Object -Unique
$batchSizes = $batchSizes | Where-Object { $_ -le $MaxBatch -and $_ -gt 0 }
if ($batchSizes.Count -eq 0) { $batchSizes = @($MaxBatch) }
$samplesRequired = ($batchSizes | Measure-Object -Sum).Sum

Write-Host "Target batch sizes  : $($batchSizes -join ', ')"
Write-Host "Required sample count: $samplesRequired"
Write-Host ""

if (-not (Test-Path $ModelPath)) {
    Write-Host "Step 1: Exporting ONNX model" -ForegroundColor Yellow
    Write-Host "----------------------------------------------------------------------"
    python export_lenet5.py `
        --checkpoint $checkpointPath `
        --format onnx `
        --output $ModelPath

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to export ONNX model." -ForegroundColor Red
        exit 1
    }
    Write-Host ""
} else {
    Write-Host "Step 1: ONNX model already exists -> $ModelPath" -ForegroundColor Yellow
    Write-Host "----------------------------------------------------------------------"
    Write-Host ""
}

$labelledCount = Get-LabeledSampleCount $binaryDir
if ($labelledCount -lt $samplesRequired) {
    $numPerClass = [Math]::Ceiling($samplesRequired / 10.0)
    if ($numPerClass -lt 5) { $numPerClass = 5 }

    Write-Host "Step 2: Exporting MNIST samples (num-per-class=$numPerClass)" -ForegroundColor Yellow
    Write-Host "----------------------------------------------------------------------"
    python export_mnist_samples.py `
        --output-dir $SamplesDir `
        --num-per-class $numPerClass `
        --formats binary png

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to export MNIST samples." -ForegroundColor Red
        exit 1
    }
    Write-Host ""
    $binaryDir = Join-Path $SamplesDir "binary"
    $labelledCount = Get-LabeledSampleCount $binaryDir
}

if ($labelledCount -lt $samplesRequired) {
    Write-Host "[ERROR] Only $labelledCount labeled binary samples available in $binaryDir." -ForegroundColor Red
    Write-Host "        Required: $samplesRequired. Please re-run export_mnist_samples.py manually."
    exit 1
}

Write-Host "Sample directory    : $binaryDir"
Write-Host "Available samples   : $labelledCount"
Write-Host ""

# Locate executable
if ($Executable) {
    $exePath = $Executable
} else {
    $candidates = @(
        (Join-Path $ScriptDir "..\..\..\build\Debug\bin\lenet5_dynamic_multi_batch.exe"),
        (Join-Path $ScriptDir "..\..\..\build\Release\bin\lenet5_dynamic_multi_batch.exe"),
        (Join-Path $ScriptDir "..\..\..\build\windows-debug\bin\lenet5_dynamic_multi_batch.exe"),
        (Join-Path $ScriptDir "..\..\..\build\windows-release\bin\lenet5_dynamic_multi_batch.exe")
    )
    $exePath = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $exePath) {
    Write-Host "[ERROR] lenet5_dynamic_multi_batch executable not found." -ForegroundColor Red
    Write-Host "        Build it with: cmake --build build --config Debug --target lenet5_dynamic_multi_batch"
    exit 1
}

Write-Host "Using executable    : $exePath" -ForegroundColor Green
Write-Host "Accuracy threshold  : $AccuracyThreshold"
Write-Host ""

$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$thresholdArg = [System.String]::Format($invariant, "{0}", $AccuracyThreshold)

Write-Host "Step 3: Running lenet5_dynamic_multi_batch" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------"
& $exePath `
    --model $ModelPath `
    --samples $binaryDir `
    --accuracy-threshold $thresholdArg `
    --max-batch $MaxBatch

$exitCode = $LASTEXITCODE
Write-Host ""

Write-Host "======================================================================" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host "[SUCCESS] Dynamic multi-batch test completed." -ForegroundColor Green
} else {
    Write-Host "[FAILED] Dynamic multi-batch test failed with exit code $exitCode." -ForegroundColor Red
}
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Model path   : $ModelPath"
Write-Host "Samples used : $binaryDir"
Write-Host "Batch sizes  : $($batchSizes -join ', ')"
Write-Host ""

Read-Host "Press Enter to exit"
exit $exitCode
