@echo off
REM Complete End-to-End Test for LeNet-5 ONNX (Windows)
REM This script should be run from models\python\lenet5 directory
REM
REM Usage: test_lenet5_onnx.bat [Debug|Release]
REM
REM Steps:
REM 1. Exports ONNX model (if not exists)
REM 2. Generates reference outputs from PyTorch
REM 3. Runs C++ inference with Mini-Infer ONNX importer
REM 4. Compares the outputs

setlocal enabledelayedexpansion

REM Default to Debug if not specified
set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Debug

echo ======================================================================
echo LeNet-5 ONNX End-to-End Test Script (Windows)
echo ======================================================================
echo.
echo Working directory: %CD%
echo Build type: %BUILD_TYPE%
echo.

REM Check if we are in the correct directory
if not exist "lenet5_model.py" (
    echo [ERROR] Script must be run from models\python\lenet5 directory
    echo.
    echo Please run:
    echo   cd models\python\lenet5
    echo   test_lenet5_onnx.bat
    echo.
    exit /b 1
)

REM Check if model checkpoint exists
if not exist "checkpoints\lenet5_best.pth" (
    echo [ERROR] Model checkpoint not found: checkpoints\lenet5_best.pth
    echo.
    echo Please train the model first:
    echo   python train_lenet5.py --epochs 10
    echo.
    exit /b 1
)

REM Check if test samples exist
if not exist "test_samples\binary" (
    echo [ERROR] Test samples not found
    echo.
    echo Please export test samples first:
    echo   python export_mnist_samples.py --num-per-class 10
    echo.
    exit /b 1
)

REM Export ONNX model if not exists
if not exist "models\lenet5.onnx" (
    echo Step 1: Exporting ONNX Model
    echo ----------------------------------------------------------------------
    python export_onnx.py ^
        --checkpoint checkpoints\lenet5_best.pth ^
        --output models\lenet5.onnx ^
        --opset-version 11
    if errorlevel 1 (
        echo [ERROR] Failed to export ONNX model
        exit /b 1
    )
    echo.
) else (
    echo Step 1: ONNX Model Already Exists
    echo ----------------------------------------------------------------------
    echo Using existing ONNX model: models\lenet5.onnx
    echo.
)

echo Step 2: Generating PyTorch Reference Outputs
echo ----------------------------------------------------------------------
python generate_reference_outputs.py ^
    --checkpoint checkpoints\lenet5_best.pth ^
    --samples-dir test_samples ^
    --output test_samples\reference_outputs.json
if errorlevel 1 (
    echo [ERROR] Failed to generate reference outputs
    exit /b 1
)
echo.

echo Step 3: Running C++ Mini-Infer ONNX Inference
echo ----------------------------------------------------------------------
..\..\..\build\%BUILD_TYPE%\bin\lenet5_onnx_test.exe ^
    models\lenet5.onnx ^
    test_samples\binary ^
    --save-outputs test_samples\minfer_onnx_outputs.json
if errorlevel 1 (
    echo.
    echo [ERROR] C++ inference failed
    echo.
    echo Note: Make sure lenet5_onnx_test is compiled:
    echo   cmake --build build --config %BUILD_TYPE% --target lenet5_onnx_test
    echo.
    exit /b 1
)
echo.

echo Step 4: Comparing Outputs
echo ----------------------------------------------------------------------
python compare_outputs.py ^
    --reference test_samples\reference_outputs.json ^
    --minfer test_samples\minfer_onnx_outputs.json ^
    --output test_samples\onnx_comparison_report.json

set COMPARE_RESULT=%errorlevel%

echo.
echo ======================================================================
if %COMPARE_RESULT%==0 (
    echo [SUCCESS] TEST PASSED: Mini-Infer ONNX matches PyTorch!
) else (
    echo [FAILED] TEST FAILED: Differences detected
)
echo ======================================================================
echo.
echo Generated files:
echo   - ONNX model: models\lenet5.onnx
echo   - Reference outputs: test_samples\reference_outputs.json
echo   - Mini-Infer ONNX outputs: test_samples\minfer_onnx_outputs.json
echo   - Comparison report: test_samples\onnx_comparison_report.json
echo.
echo View detailed report:
echo   type test_samples\onnx_comparison_report.json
echo   python -m json.tool test_samples\onnx_comparison_report.json
echo.

exit /b %COMPARE_RESULT%
