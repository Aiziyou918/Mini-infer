@echo off
REM Batch script: correctness check for optimized LeNet-5 inference
REM Usage: test_optimized_inference.bat [BUILD_DIR] [MODEL_FILE] [SAMPLES_DIR] [NUM_SAMPLES]

setlocal enabledelayedexpansion

REM Resolve defaults relative to script
set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%..\..\..\build\Debug"
set "MODEL_FILE=%SCRIPT_DIR%models\lenet5.onnx"
set "SAMPLES_DIR=%SCRIPT_DIR%test_samples"
set "NUM_SAMPLES=10"

REM Positional overrides
if not "%~1"=="" set "BUILD_DIR=%~1"
if not "%~2"=="" set "MODEL_FILE=%~2"
if not "%~3"=="" set "SAMPLES_DIR=%~3"
if not "%~4"=="" set "NUM_SAMPLES=%~4"

REM Locate executable (Debug then Release)
set "EXE_PATH=%BUILD_DIR%\bin\lenet5_optimized_inference.exe"
if not exist "%EXE_PATH%" (
    set "ALT_EXE=%SCRIPT_DIR%..\..\..\build\Release\bin\lenet5_optimized_inference.exe"
    if exist "%ALT_EXE%" (
        set "EXE_PATH=%ALT_EXE%"
    ) else (
        echo Error: lenet5_optimized_inference.exe not found.
        echo Checked:
        echo   %EXE_PATH%
        echo   %ALT_EXE%
        exit /b 1
    )
)

if not exist "%MODEL_FILE%" (
    echo Error: Model file not found: %MODEL_FILE%
    exit /b 1
)

if not exist "%SAMPLES_DIR%" (
    echo Error: Samples directory not found: %SAMPLES_DIR%
    exit /b 1
)
set "SAMPLES_BIN=%SAMPLES_DIR%\binary"
if exist "%SAMPLES_BIN%" (
    set "SAMPLES_BIN_FLAG=1"
) else (
    set "SAMPLES_BIN_FLAG=0"
)

echo ======================================================================
echo LeNet-5 Optimized End-to-End Test Script (Batch)
echo ======================================================================
echo.
echo Working directory: %cd%
for %%p in ("%EXE_PATH%") do set "BUILD_PARENT=%%~dpp"
for %%p in ("%BUILD_PARENT%") do set "BUILD_TYPE=%%~nxp"
echo Build type: %BUILD_TYPE%
echo.

echo Step 1: Generating PyTorch Reference Outputs
echo ----------------------------------------------------------------------
if "%SAMPLES_BIN_FLAG%"=="1" (
    python generate_reference_outputs.py --checkpoint checkpoints\lenet5_best.pth --samples-dir "%SAMPLES_DIR%\.." --output "%SAMPLES_DIR%\..\reference_outputs.json"
) else (
    python generate_reference_outputs.py --checkpoint checkpoints\lenet5_best.pth --samples-dir "%SAMPLES_DIR%" --output "%SAMPLES_DIR%\reference_outputs.json"
)
if not "%ERRORLEVEL%"=="0" (
    echo Error: Failed to generate reference outputs
    exit /b 1
)

echo.
echo Step 2: Running Optimized C++ Inference
echo ----------------------------------------------------------------------
echo Executable : %EXE_PATH%
echo Model      : %MODEL_FILE%
echo Samples    : %SAMPLES_DIR%
echo NumSamples : %NUM_SAMPLES%
echo.

set "OUT_JSON=%SAMPLES_DIR%\..\minfer_optimized_outputs.json"
if "%SAMPLES_BIN_FLAG%"=="0" set "OUT_JSON=%SAMPLES_DIR%\minfer_optimized_outputs.json"

if "%SAMPLES_BIN_FLAG%"=="1" (
    "%EXE_PATH%" "%MODEL_FILE%" "%SAMPLES_BIN%" %NUM_SAMPLES% --save-outputs "%OUT_JSON%" > "%TEMP%\toi_output.txt" 2>&1
) else (
    "%EXE_PATH%" "%MODEL_FILE%" "%SAMPLES_DIR%" %NUM_SAMPLES% --save-outputs "%OUT_JSON%" > "%TEMP%\toi_output.txt" 2>&1
)
set "code=%ERRORLEVEL%"
type "%TEMP%\toi_output.txt"

if not "%code%"=="0" (
    echo.
    echo [FAIL] Inference failed with exit code %code%.
    del "%TEMP%\toi_output.txt" >nul 2>&1
    exit /b %code%
)

set "ACC="
set "TOTAL="
set "CORR="
for /f "tokens=3" %%a in ('findstr /C:"Accuracy:" "%TEMP%\toi_output.txt"') do set "ACC=%%a"
for /f "tokens=3" %%a in ('findstr /C:"Total samples:" "%TEMP%\toi_output.txt"') do set "TOTAL=%%a"
for /f "tokens=3" %%a in ('findstr /C:"Correct predictions:" "%TEMP%\toi_output.txt"') do set "CORR=%%a"

set "PASS=0"
if defined ACC (
    for /f "tokens=1 delims=%" %%a in ("%ACC%") do set "ACC_NUM=%%a"
    rem Check ACC >= 99.0
    set /a ACC_INT=!ACC_NUM!
    if !ACC_INT! GEQ 99 set "PASS=1"
)
if "!PASS!"=="0" if defined TOTAL if defined CORR (
    if "!TOTAL!"=="!CORR!" set "PASS=1"
)

if "!PASS!"=="1" (
    echo.
    echo [SUCCESS] Optimized inference passed accuracy check. Accuracy=!ACC! Total=!TOTAL!
) else (
    echo.
    echo [WARN] Inference succeeded but accuracy check failed or missing metrics. Accuracy=!ACC! Total=!TOTAL!
)

echo.
echo Step 3: Comparing Outputs
echo ----------------------------------------------------------------------
if "%SAMPLES_BIN_FLAG%"=="1" (
    python compare_outputs.py --reference "%SAMPLES_DIR%\..\reference_outputs.json" --minfer "%OUT_JSON%" --output "%SAMPLES_DIR%\..\comparison_optimized.json"
) else (
    python compare_outputs.py --reference "%SAMPLES_DIR%\reference_outputs.json" --minfer "%OUT_JSON%" --output "%SAMPLES_DIR%\comparison_optimized.json"
)
set "compare=%ERRORLEVEL%"

echo.
echo ======================================================================
if "%compare%"=="0" if "!PASS!"=="1" (
    echo [SUCCESS] TEST PASSED: Optimized Mini-Infer matches PyTorch!
) else if "%compare%"=="0" (
    echo [WARN] Outputs match but runtime accuracy threshold not met.
) else (
    echo [FAILED] TEST FAILED: Differences detected
)
echo ======================================================================
echo Generated files:
if "%SAMPLES_BIN_FLAG%"=="1" (
    echo   - Reference outputs: %SAMPLES_DIR%\..\reference_outputs.json
    echo   - Mini-Infer outputs: %OUT_JSON%
    echo   - Comparison report: %SAMPLES_DIR%\..\comparison_optimized.json
) else (
    echo   - Reference outputs: %SAMPLES_DIR%\reference_outputs.json
    echo   - Mini-Infer outputs: %OUT_JSON%
    echo   - Comparison report: %SAMPLES_DIR%\comparison_optimized.json
)

if not "%compare%"=="0" (
    del "%TEMP%\toi_output.txt" >nul 2>&1
    exit /b 1
)

del "%TEMP%\toi_output.txt" >nul 2>&1

endlocal

