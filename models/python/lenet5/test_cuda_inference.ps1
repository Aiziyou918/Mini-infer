# PowerShell script: CUDA GPU inference test for LeNet-5
# Usage:
#   ./test_cuda_inference.ps1 [-BuildDir <path>] [-ModelFile <path>] [-SamplesDir <path>] [-NumSamples <int>] [-CompareCPU]

param(
    [string]$BuildDir = "",
    [string]$ModelFile = "",
    [string]$SamplesDir = "",
    [int]$NumSamples = 50,
    [switch]$CompareCPU
)

$ScriptDir = Split-Path -Parent $PSCommandPath
if (-not $BuildDir)   { $BuildDir   = Join-Path $ScriptDir "../../../build/Debug" }
if (-not $ModelFile)  { $ModelFile  = Join-Path $ScriptDir "models/lenet5.onnx" }
if (-not $SamplesDir) { $SamplesDir = Join-Path $ScriptDir "test_samples" }

$DebugExe   = Join-Path $BuildDir "bin/lenet5_cuda_inference.exe"
$ReleaseExe = Join-Path $ScriptDir "../../../build/Release/bin/lenet5_cuda_inference.exe"
$ExePath = $DebugExe
if (-not (Test-Path $ExePath) -and (Test-Path $ReleaseExe)) { $ExePath = $ReleaseExe }

if (-not (Test-Path $ExePath)) {
    Write-Host "Error: lenet5_cuda_inference.exe not found." -ForegroundColor Red
    Write-Host "Checked:" -ForegroundColor Yellow
    Write-Host "  $DebugExe"
    Write-Host "  $ReleaseExe"
    Write-Host ""
    Write-Host "Make sure you built with CUDA support enabled:" -ForegroundColor Yellow
    Write-Host "  cmake -DMINI_INFER_ENABLE_CUDA=ON ..."
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
Write-Host "LeNet-5 CUDA GPU Inference Test Script (PowerShell)"
Write-Host "======================================================================"
Write-Host ""
Write-Host "Working directory: $(Get-Location)"
$buildType = Split-Path -Leaf (Split-Path -Parent $ExePath)
Write-Host "Build type: $buildType"
Write-Host ""

# Step 1: Run CUDA GPU inference
Write-Host "Step 1: Running CUDA GPU Inference" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------"
Write-Host "Executable : $ExePath"
Write-Host "Model      : $ModelFile"
Write-Host "Samples    : $SamplesBin"
Write-Host "NumSamples : $NumSamples"
Write-Host ""

$cmdArgs = @($ModelFile, $SamplesBin, $NumSamples)
if ($CompareCPU) {
    $cmdArgs += "--compare-cpu"
}

$output = (& $ExePath @cmdArgs 2>&1 | Out-String)
$code = $LASTEXITCODE
Write-Host $output

if ($code -ne 0) {
    Write-Host "`n[FAIL] CUDA inference failed with exit code $code." -ForegroundColor Red
    exit $code
}

# Parse results
$accMatch   = [regex]::Match($output, "Accuracy:\s+([\d\.]+)\s*%")
$corrMatch  = [regex]::Match($output, "Correct predictions:\s+(\d+)")
$totalMatch = [regex]::Match($output, "Total samples:\s+(\d+)")
$avgTimeMatch = [regex]::Match($output, "Average time per sample:\s+([\d\.]+)\s*ms")
$throughputMatch = [regex]::Match($output, "Throughput:\s+([\d\.]+)\s*samples/sec")

$pass = $false
if ($accMatch.Success -and $totalMatch.Success) {
    $acc = [double]$accMatch.Groups[1].Value
    $total = [int]$totalMatch.Groups[1].Value
    $pass = ($acc -ge 99.0) -or ($corrMatch.Success -and ([int]$corrMatch.Groups[1].Value -eq $total))
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
if ($pass) {
    Write-Host "[SUCCESS] CUDA GPU inference passed!" -ForegroundColor Green
    if ($accMatch.Success) {
        Write-Host "  Accuracy: $($accMatch.Groups[1].Value)%" -ForegroundColor Green
    }
    if ($totalMatch.Success) {
        Write-Host "  Total samples: $($totalMatch.Groups[1].Value)" -ForegroundColor Green
    }
    if ($avgTimeMatch.Success) {
        Write-Host "  Avg time/sample: $($avgTimeMatch.Groups[1].Value) ms" -ForegroundColor Green
    }
    if ($throughputMatch.Success) {
        Write-Host "  Throughput: $($throughputMatch.Groups[1].Value) samples/sec" -ForegroundColor Green
    }
} else {
    Write-Host "[WARN] Inference succeeded but accuracy check failed or missing metrics." -ForegroundColor Yellow
}
Write-Host "======================================================================" -ForegroundColor Cyan

if (-not $pass) { exit 1 }
