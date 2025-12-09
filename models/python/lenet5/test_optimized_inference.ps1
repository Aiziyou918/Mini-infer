# PowerShell script: correctness check for optimized LeNet-5 inference
# Usage:
#   ./test_optimized_inference.ps1 [-BuildDir <path>] [-ModelFile <path>] [-SamplesDir <path>] [-NumSamples <int>]

param(
    [string]$BuildDir = "",
    [string]$ModelFile = "",
    [string]$SamplesDir = "",
    [int]$NumSamples = 10
)

$ScriptDir = Split-Path -Parent $PSCommandPath
if (-not $BuildDir)   { $BuildDir   = Join-Path $ScriptDir "../../../build/Debug" }
if (-not $ModelFile)  { $ModelFile  = Join-Path $ScriptDir "models/lenet5.onnx" }
if (-not $SamplesDir) { $SamplesDir = Join-Path $ScriptDir "test_samples" }  # expect binary under it

$DebugExe   = Join-Path $BuildDir "bin/lenet5_optimized_inference.exe"
$ReleaseExe = Join-Path $ScriptDir "../../../build/Release/bin/lenet5_optimized_inference.exe"
$ExePath = $DebugExe
if (-not (Test-Path $ExePath) -and (Test-Path $ReleaseExe)) { $ExePath = $ReleaseExe }

if (-not (Test-Path $ExePath)) {
    Write-Host "Error: lenet5_optimized_inference.exe not found." -ForegroundColor Red
    Write-Host "Checked:" -ForegroundColor Yellow
    Write-Host "  $DebugExe"
    Write-Host "  $ReleaseExe"
    exit 1
}
if (-not (Test-Path $ModelFile)) {
    Write-Host "Error: Model file not found: $ModelFile" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $SamplesDir)) {
    Write-Host "Error: Samples directory not found: $SamplesDir" -ForegroundColor Red
    exit 1
}
$SamplesBin = Join-Path $SamplesDir "binary"
if (-not (Test-Path $SamplesBin)) { $SamplesBin = $SamplesDir }

Write-Host "======================================================================"
Write-Host "LeNet-5 Optimized End-to-End Test Script (PowerShell)"
Write-Host "======================================================================"
Write-Host ""
Write-Host "Working directory: $(Get-Location)"
$buildType = Split-Path -Leaf (Split-Path -Parent $ExePath)
Write-Host "Build type: $buildType"
Write-Host ""

# Step 1: generate reference outputs (PyTorch)
Write-Host "Step 1: Generating PyTorch Reference Outputs" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------"
python generate_reference_outputs.py `
    --checkpoint checkpoints\lenet5_best.pth `
    --samples-dir "$SamplesDir" `
    --output "$SamplesDir\reference_outputs.json"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to generate reference outputs" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: run optimized C++ inference (only accuracy available)
Write-Host "Step 2: Running Optimized C++ Inference" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------"
Write-Host "Executable : $ExePath"
Write-Host "Model      : $ModelFile"
Write-Host "Samples    : $SamplesBin"
Write-Host "NumSamples : $NumSamples"
Write-Host ""

$outputPath = Join-Path $SamplesDir "minfer_optimized_outputs.json"
$output = (& $ExePath $ModelFile $SamplesBin $NumSamples --save-outputs $outputPath 2>&1 | Out-String)
$code = $LASTEXITCODE
Write-Host $output
if ($code -ne 0) {
    Write-Host "`n[FAIL] Inference failed with exit code $code." -ForegroundColor Red
    exit $code
}

$accMatch   = [regex]::Match($output, "Accuracy:\s+([\d\.]+)%")
$corrMatch  = [regex]::Match($output, "Correct predictions:\s+(\d+)")
$totalMatch = [regex]::Match($output, "Total samples:\s+(\d+)")
$pass = $false
if ($accMatch.Success -and $totalMatch.Success) {
    $acc = [double]$accMatch.Groups[1].Value
    $total = [int]$totalMatch.Groups[1].Value
    $pass = ($acc -ge 99.0) -or ($corrMatch.Success -and ([int]$corrMatch.Groups[1].Value -eq $total))
    if ($pass) {
        Write-Host "`n[SUCCESS] Optimized inference passed accuracy check. Accuracy=${acc}%, Total=$total" -ForegroundColor Green
    } else {
        Write-Host "`n[FAIL] Accuracy is lower than expected. Accuracy=${acc}%, Total=$total" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n[WARN] Accuracy metrics not found in output; only runtime succeeded." -ForegroundColor Yellow
    $pass = $true
}

Write-Host ""
Write-Host "Step 3: Comparing Outputs" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------"
python compare_outputs.py `
    --reference (Join-Path $SamplesDir "reference_outputs.json") `
    --minfer $outputPath `
    --output (Join-Path $SamplesDir "comparison_optimized.json")

$compareResult = $LASTEXITCODE

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
if ($compareResult -eq 0 -and $pass) {
    Write-Host "[SUCCESS] TEST PASSED: Optimized Mini-Infer matches PyTorch!" -ForegroundColor Green
} elseif ($compareResult -eq 0) {
    Write-Host "[WARN] Outputs match but accuracy threshold not met in runtime log." -ForegroundColor Yellow
} else {
    Write-Host "[FAILED] TEST FAILED: Differences detected" -ForegroundColor Red
}
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Generated files:"
Write-Host "  - Reference outputs: $(Join-Path $SamplesDir "reference_outputs.json")"
Write-Host "  - Mini-Infer outputs: $outputPath"
Write-Host "  - Comparison report: $(Join-Path $SamplesDir "comparison_optimized.json")"

if (-not $pass -or $compareResult -ne 0) { exit 1 }

