@echo off
REM Complete End-to-End Test for LeNet-5
REM This script should be run from models/python/lenet5 directory
REM 
REM Usage: test_lenet5.bat
REM
REM Steps:
REM 1. Generates reference outputs from PyTorch
REM 2. Runs C++ inference with Mini-Infer
REM 3. Compares the outputs

setlocal enabledelayedexpansion

echo ======================================================================
echo LeNet-5 End-to-End Test Script
echo ======================================================================
echo.
echo Working directory: %CD%
echo.

REM Check if we are in the correct directory
if not exist "lenet5_model.py" (
    echo Error: Script must be run from models/python/lenet5 directory
    echo.
    echo Please run:
    echo   cd models\python\lenet5
    echo   test_lenet5.bat
    echo.
    pause
    exit /b 1
)

REM Check if model checkpoint exists
if not exist "checkpoints\lenet5_best.pth" (
    echo Error: Model checkpoint not found: checkpoints\lenet5_best.pth
    echo.
    echo Please train the model first:
    echo   python train_lenet5.py --epochs 10
    echo.
    pause
    exit /b 1
)

REM Check if test samples exist
if not exist "test_samples\binary" (
    echo Error: Test samples not found
    echo.
    echo Please export test samples first:
    echo   python export_mnist_samples.py --num-per-class 10
    echo.
    pause
    exit /b 1
)

echo Step 1: Generating PyTorch Reference Outputs
echo ----------------------------------------------------------------------
python generate_reference_outputs.py ^
    --checkpoint checkpoints\lenet5_best.pth ^
    --samples-dir test_samples ^
    --output test_samples\reference_outputs.json

if %errorlevel% neq 0 (
    echo Error: Failed to generate reference outputs
    pause
    exit /b 1
)
echo.

echo Step 2: Running C++ Mini-Infer Inference
echo ----------------------------------------------------------------------
..\..\..\build\windows-debug\bin\lenet5_inference.exe ^
    weights ^
    test_samples\binary ^
    --save-outputs test_samples\minfer_outputs.json

if %errorlevel% neq 0 (
    echo Error: Failed to run C++ inference
    echo.
    echo Note: Make sure lenet5_inference is compiled:
    echo   cmake --build build --config Debug --target lenet5_inference
    echo.
    pause
    exit /b 1
)
echo.

echo Step 3: Comparing Outputs
echo ----------------------------------------------------------------------
python compare_outputs.py ^
    --reference test_samples\reference_outputs.json ^
    --minfer test_samples\minfer_outputs.json ^
    --output test_samples\comparison_report.json

set COMPARE_RESULT=%errorlevel%

echo.
echo ======================================================================
if %COMPARE_RESULT% equ 0 (
    echo [SUCCESS] TEST PASSED: Mini-Infer matches PyTorch!
) else (
    echo [FAILED] TEST FAILED: Differences detected
)
echo ======================================================================
echo.
echo Generated files:
echo   - Reference outputs: test_samples\reference_outputs.json
echo   - Mini-Infer outputs: test_samples\minfer_outputs.json
echo   - Comparison report: test_samples\comparison_report.json
echo.
echo View detailed report:
echo   type test_samples\comparison_report.json
echo.

pause
exit /b %COMPARE_RESULT%
