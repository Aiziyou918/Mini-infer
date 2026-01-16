# ============================================================================
# LeNet-5 Optimized Inference Test with Memory Planning (PowerShell)
# This script tests the optimized inference and compares with PyTorch reference
# ============================================================================

param(
    [string]$BuildType = "Debug"
)

Write-Host "========================================"
Write-Host "LeNet-5 Optimized Inference Test"
Write-Host "Build Type: $BuildType"
Write-Host "========================================"
Write-Host ""

# Configuration
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$MODEL_PATH = Join-Path $SCRIPT_DIR "models\lenet5.onnx"
$SAMPLES_DIR = Join-Path $SCRIPT_DIR "test_samples"

# Find executable
$candidates = @(
    (Join-Path $SCRIPT_DIR "..\..\..\build\Debug\bin\lenet5_optimized_with_memory_planning.exe"),
    (Join-Path $SCRIPT_DIR "..\..\..\build\Release\bin\lenet5_optimized_with_memory_planning.exe"),
    (Join-Path $SCRIPT_DIR "..\..\..\build\bin\lenet5_optimized_with_memory_planning.exe")
)

$EXECUTABLE = $null
foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
        $EXECUTABLE = $candidate
        break
    }
}

# Check executable
if (-not $EXECUTABLE) {
    Write-Host "[ERROR] Executable not found" -ForegroundColor Red
    Write-Host "Checked:"
    foreach ($candidate in $candidates) {
        Write-Host "  $candidate"
    }
    Write-Host ""
    Write-Host "Please build the project first:"
    Write-Host "  cmake --build build/Debug --parallel"
    exit 1
}

Write-Host "Using executable: $EXECUTABLE"
Write-Host ""

# Check model
if (-not (Test-Path $MODEL_PATH)) {
    Write-Host "[ERROR] Model not found: $MODEL_PATH" -ForegroundColor Red
    exit 1
}

# Check samples
if (-not (Test-Path $SAMPLES_DIR)) {
    Write-Host "[ERROR] Samples directory not found: $SAMPLES_DIR" -ForegroundColor Red
    exit 1
}

# Use binary samples if available
$SAMPLES_BIN = Join-Path $SAMPLES_DIR "binary"
if (Test-Path $SAMPLES_BIN) {
    $SAMPLES_DIR = $SAMPLES_BIN
}

Write-Host "========================================"
Write-Host "Step 1: Generate PyTorch Reference"
Write-Host "========================================"
Write-Host ""

python generate_reference_outputs.py `
    --checkpoint checkpoints\lenet5_best.pth `
    --samples-dir test_samples `
    --output test_samples\reference_outputs.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to generate reference outputs" -ForegroundColor Red
    exit 1
}
Write-Host ""

Write-Host "========================================"
Write-Host "Step 2: Run Optimized Inference (WITH Memory Planning)"
Write-Host "========================================"
Write-Host ""

$OUTPUT_WITH_MEM = Join-Path $SCRIPT_DIR "test_samples\optimized_memory_outputs.json"
& $EXECUTABLE --model $MODEL_PATH --samples $SAMPLES_DIR --save-outputs $OUTPUT_WITH_MEM

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Optimized inference failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

Write-Host "========================================"
Write-Host "Step 3: Run Optimized Inference (WITHOUT Memory Planning)"
Write-Host "========================================"
Write-Host ""

$OUTPUT_NO_MEM = Join-Path $SCRIPT_DIR "test_samples\optimized_no_memory_outputs.json"
& $EXECUTABLE --model $MODEL_PATH --samples $SAMPLES_DIR --no-memory-planning --save-outputs $OUTPUT_NO_MEM

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Optimized inference failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

Write-Host "========================================"
Write-Host "Step 4: Compare with PyTorch Reference"
Write-Host "========================================"
Write-Host ""

python compare_outputs.py `
    --reference test_samples\reference_outputs.json `
    --minfer $OUTPUT_WITH_MEM `
    --output test_samples\optimized_comparison_report.json

$COMPARE_RESULT = $LASTEXITCODE
Write-Host ""

Write-Host "========================================"
Write-Host "Step 5: Memory Usage Comparison"
Write-Host "========================================"
Write-Host ""

python compare_memory_usage.py `
    $OUTPUT_NO_MEM `
    $OUTPUT_WITH_MEM

Write-Host ""
Write-Host "========================================"
Write-Host "Test Summary"
Write-Host "========================================"
Write-Host ""

if ($COMPARE_RESULT -eq 0) {
    Write-Host "[SUCCESS] Optimized inference matches PyTorch reference!" -ForegroundColor Green
    Write-Host ""
    Write-Host "[PASS] All tests passed!" -ForegroundColor Green
} else {
    Write-Host "[FAILED] Optimized inference does NOT match PyTorch reference" -ForegroundColor Red
    Write-Host ""
    Write-Host "[FAIL] Tests failed" -ForegroundColor Red
}

Write-Host "========================================"

exit $COMPARE_RESULT
