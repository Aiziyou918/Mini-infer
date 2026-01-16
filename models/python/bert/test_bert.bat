@echo off
REM
REM BERT Inference Test Script (Windows Batch)
REM 对比 Mini-Infer 和 PyTorch 的 BERT 推理结果
REM
REM Usage:
REM   test_bert.bat                        使用默认配置
REM   test_bert.bat --verbose              显示详细对比
REM   test_bert.bat --regenerate           重新生成 PyTorch 参考输出
REM   test_bert.bat --build-type Release   使用 Release 构建
REM

setlocal enabledelayedexpansion

REM 脚本所在目录
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM 项目根目录
pushd "%SCRIPT_DIR%\..\..\..\"
set "PROJECT_ROOT=%CD%"
popd

REM 默认配置
set "BUILD_TYPE=Debug"
set "NUM_SAMPLES=10"
set "VERBOSE="
set "REGENERATE="
set "CONDA_ENV=mini-infer"

REM 解析命令行参数
:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--build-type" (
    set "BUILD_TYPE=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--num-samples" (
    set "NUM_SAMPLES=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--verbose" (
    set "VERBOSE=--verbose"
    shift
    goto :parse_args
)
if /i "%~1"=="-v" (
    set "VERBOSE=--verbose"
    shift
    goto :parse_args
)
if /i "%~1"=="--regenerate" (
    set "REGENERATE=--regenerate-reference"
    shift
    goto :parse_args
)
if /i "%~1"=="--conda-env" (
    set "CONDA_ENV=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
echo Unknown option: %~1
exit /b 1

:show_help
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --build-type TYPE    Build type: Debug or Release (default: Debug)
echo   --num-samples N      Number of samples to test (default: 10)
echo   --verbose, -v        Show detailed comparison
echo   --regenerate         Regenerate PyTorch reference outputs
echo   --conda-env ENV      Conda environment name (default: mini-infer)
echo   --help, -h           Show this help message
exit /b 0

:done_args

REM 路径配置
set "MINI_INFER_BIN=%PROJECT_ROOT%\build\%BUILD_TYPE%\bin\bert_inference.exe"
set "MODEL_PATH=%SCRIPT_DIR%\models\bert_tiny.onnx"
set "SAMPLES_DIR=%SCRIPT_DIR%\test_samples"

echo ======================================================================
echo BERT Inference Test - Mini-Infer vs PyTorch
echo ======================================================================
echo.
echo Configuration:
echo   Project root: %PROJECT_ROOT%
echo   Build type: %BUILD_TYPE%
echo   Mini-Infer binary: %MINI_INFER_BIN%
echo   Model: %MODEL_PATH%
echo   Samples: %SAMPLES_DIR%
echo   Num samples: %NUM_SAMPLES%
echo.

REM 检查 Mini-Infer 二进制文件
if not exist "%MINI_INFER_BIN%" (
    echo [ERROR] Mini-Infer binary not found: %MINI_INFER_BIN%
    echo.
    echo Please build the project first:
    echo   cmake --build build/%BUILD_TYPE% --parallel
    exit /b 1
)

REM 检查模型文件
if not exist "%MODEL_PATH%" (
    echo [ERROR] Model not found: %MODEL_PATH%
    echo.
    echo Please export the model first:
    echo   cd %SCRIPT_DIR% ^&^& python export_bert.py
    exit /b 1
)

REM 检查样本目录
if not exist "%SAMPLES_DIR%" (
    echo [ERROR] Samples directory not found: %SAMPLES_DIR%
    echo.
    echo Please generate samples first:
    echo   cd %SCRIPT_DIR% ^&^& python export_bert_samples.py
    exit /b 1
)

REM 激活 conda 环境（如果可用）
where conda >nul 2>&1
if %errorlevel% equ 0 (
    echo Activating conda environment: %CONDA_ENV%
    call conda activate %CONDA_ENV% 2>nul
    if errorlevel 1 (
        echo [WARN] Could not activate conda environment: %CONDA_ENV%
        echo        Continuing with current Python environment...
    )
    echo.
)

REM 运行对比测试
echo Running comparison test...
echo.

python "%SCRIPT_DIR%\test_bert_comparison.py" ^
    --mini-infer-bin "%MINI_INFER_BIN%" ^
    --model "%MODEL_PATH%" ^
    --samples-dir "%SAMPLES_DIR%" ^
    --num-samples %NUM_SAMPLES% ^
    %VERBOSE% ^
    %REGENERATE%

set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% equ 0 (
    echo [SUCCESS] BERT inference test completed successfully!
) else (
    echo [FAILED] BERT inference test failed!
)

exit /b %EXIT_CODE%
