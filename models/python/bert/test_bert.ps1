#
# BERT Inference Test Script (PowerShell)
# 对比 Mini-Infer 和 PyTorch 的 BERT 推理结果
#
# Usage:
#   .\test_bert.ps1                        # 使用默认配置
#   .\test_bert.ps1 -Verbose               # 显示详细对比
#   .\test_bert.ps1 -Regenerate            # 重新生成 PyTorch 参考输出
#   .\test_bert.ps1 -BuildType Release     # 使用 Release 构建
#

param(
    [ValidateSet("Debug", "Release")]
    [string]$BuildType = "Debug",

    [int]$NumSamples = 10,

    [switch]$Verbose,

    [switch]$Regenerate,

    [string]$CondaEnv = "mini-infer",

    [switch]$Help
)

# 显示帮助
if ($Help) {
    Write-Host "Usage: .\test_bert.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -BuildType TYPE    Build type: Debug or Release (default: Debug)"
    Write-Host "  -NumSamples N      Number of samples to test (default: 10)"
    Write-Host "  -Verbose           Show detailed comparison"
    Write-Host "  -Regenerate        Regenerate PyTorch reference outputs"
    Write-Host "  -CondaEnv ENV      Conda environment name (default: mini-infer)"
    Write-Host "  -Help              Show this help message"
    exit 0
}

# 脚本所在目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path "$ScriptDir\..\..\..").Path

# 路径配置
$MiniInferBin = Join-Path $ProjectRoot "build\$BuildType\bin\bert_inference.exe"
$ModelPath = Join-Path $ScriptDir "models\bert_tiny.onnx"
$SamplesDir = Join-Path $ScriptDir "test_samples"

Write-Host "======================================================================"
Write-Host "BERT Inference Test - Mini-Infer vs PyTorch"
Write-Host "======================================================================"
Write-Host ""
Write-Host "Configuration:"
Write-Host "  Project root: $ProjectRoot"
Write-Host "  Build type: $BuildType"
Write-Host "  Mini-Infer binary: $MiniInferBin"
Write-Host "  Model: $ModelPath"
Write-Host "  Samples: $SamplesDir"
Write-Host "  Num samples: $NumSamples"
Write-Host ""

# 检查 Mini-Infer 二进制文件
if (-not (Test-Path $MiniInferBin)) {
    Write-Host "[ERROR] Mini-Infer binary not found: $MiniInferBin" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please build the project first:"
    Write-Host "  cmake --build build/$BuildType --parallel"
    exit 1
}

# 检查模型文件
if (-not (Test-Path $ModelPath)) {
    Write-Host "[ERROR] Model not found: $ModelPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please export the model first:"
    Write-Host "  cd $ScriptDir; python export_bert.py"
    exit 1
}

# 检查样本目录
if (-not (Test-Path $SamplesDir)) {
    Write-Host "[ERROR] Samples directory not found: $SamplesDir" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please generate samples first:"
    Write-Host "  cd $ScriptDir; python export_bert_samples.py"
    exit 1
}

# 激活 conda 环境（如果可用）
$condaPath = Get-Command conda -ErrorAction SilentlyContinue
if ($condaPath) {
    Write-Host "Activating conda environment: $CondaEnv"
    try {
        # 初始化 conda for PowerShell
        $condaHook = & conda shell.powershell hook 2>$null
        if ($condaHook) {
            Invoke-Expression $condaHook
            conda activate $CondaEnv 2>$null
        }
    }
    catch {
        Write-Host "[WARN] Could not activate conda environment: $CondaEnv" -ForegroundColor Yellow
        Write-Host "       Continuing with current Python environment..."
    }
    Write-Host ""
}

# 构建参数列表
$pythonArgs = @(
    (Join-Path $ScriptDir "test_bert_comparison.py"),
    "--mini-infer-bin", $MiniInferBin,
    "--model", $ModelPath,
    "--samples-dir", $SamplesDir,
    "--num-samples", $NumSamples
)

if ($Verbose) {
    $pythonArgs += "--verbose"
}

if ($Regenerate) {
    $pythonArgs += "--regenerate-reference"
}

# 运行对比测试
Write-Host "Running comparison test..."
Write-Host ""

& python $pythonArgs
$ExitCode = $LASTEXITCODE

Write-Host ""
if ($ExitCode -eq 0) {
    Write-Host "[SUCCESS] BERT inference test completed successfully!" -ForegroundColor Green
}
else {
    Write-Host "[FAILED] BERT inference test failed!" -ForegroundColor Red
}

exit $ExitCode
