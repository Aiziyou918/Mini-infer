@echo off
REM ============================================================================
REM LeNet-5 Optimized Inference Test with Memory Planning
REM This script tests the optimized inference (graph optimization + memory planning)
REM and compares results with PyTorch reference outputs
REM ============================================================================

setlocal enabledelayedexpansion

REM Default to Debug if not specified
set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Debug

echo ========================================
echo LeNet-5 Optimized Inference Test
echo Build Type: %BUILD_TYPE%
echo ========================================
echo.

REM Configuration (relative to script dir)
set "SCRIPT_DIR=%~dp0"
set "MODEL_PATH=%SCRIPT_DIR%models\lenet5.onnx"
set "SAMPLES_DIR=%SCRIPT_DIR%test_samples"
set "EXEC_DEBUG=%SCRIPT_DIR%..\..\..\build\Debug\bin\lenet5_optimized_with_memory_planning.exe"
set "EXEC_RELEASE=%SCRIPT_DIR%..\..\..\build\Release\bin\lenet5_optimized_with_memory_planning.exe"

REM Select executable based on build type
if /i "%BUILD_TYPE%"=="Release" (
    set "EXECUTABLE=%EXEC_RELEASE%"
) else (
    set "EXECUTABLE=%EXEC_DEBUG%"
)

REM Fallback if specified build type doesn't exist
if not exist "%EXECUTABLE%" (
    if exist "%EXEC_DEBUG%" (
        set "EXECUTABLE=%EXEC_DEBUG%"
        set "BUILD_TYPE=Debug"
    ) else if exist "%EXEC_RELEASE%" (
        set "EXECUTABLE=%EXEC_RELEASE%"
        set "BUILD_TYPE=Release"
    )
)

REM Check if executable exists
if not exist "%EXECUTABLE%" (
    echo [ERROR] Executable not found in Debug/Release.
    echo Checked:
    echo   %EXEC_DEBUG%
    echo   %EXEC_RELEASE%
    echo.
    echo Please build the project first:
    echo   cd build
    echo   cmake --build . --config Debug
    exit /b 1
)

echo Using executable: %EXECUTABLE%
echo.

REM Check if model exists
if not exist "%MODEL_PATH%" (
    echo [ERROR] Model not found: %MODEL_PATH%
    echo.
    echo Please export ONNX model first:
    echo   python export_onnx.py --checkpoint checkpoints\lenet5_best.pth --output models\lenet5.onnx
    exit /b 1
)

REM Check if samples directory exists
if not exist "%SAMPLES_DIR%" (
    echo [ERROR] Samples directory not found: %SAMPLES_DIR%
    echo.
    echo Please export test samples first:
    echo   python export_mnist_samples.py --num-per-class 10
    exit /b 1
)

REM Use binary samples if available
set "SAMPLES_BIN=%SAMPLES_DIR%\binary"
if exist "%SAMPLES_BIN%" set "SAMPLES_DIR=%SAMPLES_BIN%"

echo ========================================
echo Step 1: Generate PyTorch Reference Outputs
echo ========================================
echo.

REM Generate reference outputs from PyTorch
python generate_reference_outputs.py ^
    --checkpoint checkpoints\lenet5_best.pth ^
    --samples-dir test_samples ^
    --output test_samples\reference_outputs.json

if errorlevel 1 (
    echo [ERROR] Failed to generate reference outputs
    exit /b 1
)
echo.

echo ========================================
echo Step 2: Run Optimized Inference (WITH Memory Planning)
echo ========================================
echo.

set "OUTPUT_WITH_MEM=%SCRIPT_DIR%test_samples\optimized_memory_outputs.json"
"%EXECUTABLE%" --model "%MODEL_PATH%" --samples "%SAMPLES_DIR%" --save-outputs "%OUTPUT_WITH_MEM%"

if "%ERRORLEVEL%" neq "0" (
    echo [ERROR] Optimized inference with memory planning failed
    exit /b 1
)
echo.

echo ========================================
echo Step 3: Run Optimized Inference (WITHOUT Memory Planning)
echo ========================================
echo.

set "OUTPUT_NO_MEM=%SCRIPT_DIR%test_samples\optimized_no_memory_outputs.json"
"%EXECUTABLE%" --model "%MODEL_PATH%" --samples "%SAMPLES_DIR%" --no-memory-planning --save-outputs "%OUTPUT_NO_MEM%"

if "%ERRORLEVEL%" neq "0" (
    echo [ERROR] Optimized inference without memory planning failed
    exit /b 1
)
echo.

echo ========================================
echo Step 4: Compare with PyTorch Reference
echo ========================================
echo.

REM Compare optimized inference (with memory planning) against PyTorch
python compare_outputs.py ^
    --reference test_samples\reference_outputs.json ^
    --minfer "%OUTPUT_WITH_MEM%" ^
    --output test_samples\optimized_comparison_report.json

set COMPARE_RESULT=%errorlevel%
echo.

echo ========================================
echo Step 5: Memory Usage Comparison
echo ========================================
echo.

REM Compare memory usage between with/without memory planning
python compare_memory_usage.py ^
    "%OUTPUT_NO_MEM%" ^
    "%OUTPUT_WITH_MEM%"

echo.

echo ========================================
echo Test Summary
echo ========================================
echo.

if "%COMPARE_RESULT%"=="0" (
    echo [SUCCESS] Optimized inference matches PyTorch reference!
    echo.
    echo Generated files:
    echo   - PyTorch reference:           test_samples\reference_outputs.json
    echo   - Optimized (with memory):     test_samples\optimized_memory_outputs.json
    echo   - Optimized (without memory):  test_samples\optimized_no_memory_outputs.json
    echo   - Comparison report:           test_samples\optimized_comparison_report.json
    echo.
    echo Key findings:
    findstr /C:"accuracy" "!OUTPUT_WITH_MEM!"
    findstr /C:"memory_stats" "!OUTPUT_WITH_MEM!"
    echo.
    echo [PASS] All tests passed!
) else (
    echo [FAILED] Optimized inference does NOT match PyTorch reference
    echo.
    echo Please check:
    echo   - Comparison report: test_samples\optimized_comparison_report.json
    echo   - Optimized outputs: test_samples\optimized_memory_outputs.json
    echo   - Reference outputs: test_samples\reference_outputs.json
    echo.
    echo [FAIL] Tests failed
)

echo ========================================

exit /b %COMPARE_RESULT%
