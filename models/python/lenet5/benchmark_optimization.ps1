# PowerShell script: benchmark optimized vs non-optimized inference
# Usage:
#   ./benchmark_optimization.ps1 [-BuildDir <path>] [-ModelFile <path>] [-SamplesDir <path>] [-NumSamples <int>] [-Iterations <int>]

param(
    [string]$BuildDir = "",
    [string]$ModelFile = "",
    [string]$SamplesDir = "",
    [int]$NumSamples = 100,
    [int]$Iterations = 20,
    [int]$Warmup = 3
)

$ScriptDir = Split-Path -Parent $PSCommandPath
if (-not $BuildDir)   { $BuildDir   = Join-Path $ScriptDir "../../../build/Debug" }
if (-not $ModelFile)  { $ModelFile  = Join-Path $ScriptDir "models/lenet5.onnx" }
if (-not $SamplesDir) { $SamplesDir = Join-Path $ScriptDir "test_samples/binary" }

$DebugExe   = Join-Path $BuildDir "bin/lenet5_optimized_inference.exe"
$ReleaseExe = Join-Path $ScriptDir "../../../build/Release/bin/lenet5_optimized_inference.exe"
$ExePath = $DebugExe
if (-not (Test-Path $ExePath) -and (Test-Path $ReleaseExe)) {
    $ExePath = $ReleaseExe
}
if (-not (Test-Path $ExePath)) {
    Write-Host "Error: lenet5_optimized_inference.exe not found." -ForegroundColor Red
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

$origTimes = @()
$optTimes  = @()

Write-Host "========================================"
Write-Host "LeNet-5 Optimization Benchmark"
Write-Host "========================================"
Write-Host "Executable : $ExePath"
Write-Host "Model      : $ModelFile"
Write-Host "Samples    : $SamplesDir"
Write-Host "NumSamples : $NumSamples"
Write-Host "Iterations : $Iterations"
Write-Host "Warmup     : $Warmup"
Write-Host ""

# Warmup (not measured)
if ($Warmup -gt 0) {
    for ($w = 1; $w -le $Warmup; $w++) {
        & $ExePath $ModelFile $SamplesDir $NumSamples --no-optimization *> $null
        & $ExePath $ModelFile $SamplesDir $NumSamples *> $null
    }
}

for ($i = 1; $i -le $Iterations; $i++) {
    Write-Host "Iteration $i / $Iterations"
    # No optimization
    $out1 = & $ExePath $ModelFile $SamplesDir $NumSamples --no-optimization 2>&1
    $out1Text = ($out1 -join "`n")
    if ($out1Text -match "Average time per sample:\s+([\d.]+)\s+ms") {
        $t1 = [double]$matches[1]; $origTimes += $t1
        Write-Host "  Original : $t1 ms"
    } else {
        Write-Host "  Original : failed to parse timing" -ForegroundColor Yellow
    }
    # Optimization
    $out2 = & $ExePath $ModelFile $SamplesDir $NumSamples 2>&1
    $out2Text = ($out2 -join "`n")
    if ($out2Text -match "Average time per sample:\s+([\d.]+)\s+ms") {
        $t2 = [double]$matches[1]; $optTimes += $t2
        Write-Host "  Optimized: $t2 ms"
    } else {
        Write-Host "  Optimized: failed to parse timing" -ForegroundColor Yellow
    }
    Write-Host ""
}

if ($origTimes.Count -eq 0 -or $optTimes.Count -eq 0) {
    Write-Host "No timing data collected." -ForegroundColor Red
    exit 1
}

$avgOrig = ($origTimes | Measure-Object -Average).Average
$avgOpt  = ($optTimes  | Measure-Object -Average).Average
$speedup = $avgOrig / $avgOpt
$improve = (($avgOrig - $avgOpt) / $avgOrig) * 100

Write-Host "========================================"
Write-Host "Results (averaged)"
Write-Host "========================================"
Write-Host ("Original  : {0:F3} ms  ({1:F1} fps)" -f $avgOrig, (1000/$avgOrig))
Write-Host ("Optimized : {0:F3} ms  ({1:F1} fps)" -f $avgOpt,  (1000/$avgOpt))
Write-Host ("Speedup   : {0:F2}x" -f $speedup)
Write-Host ("Time drop : {0:F1}%" -f $improve)

if ($speedup -gt 1.05) {
    Write-Host "PASS: Optimization is effective." -ForegroundColor Green
} elseif ($speedup -gt 0.95) {
    Write-Host "WARN: Optimization impact is minimal." -ForegroundColor Yellow
} else {
    Write-Host "FAIL: Optimization degraded performance." -ForegroundColor Red
}

