@echo off
REM Batch script: CUDA GPU inference test for LeNet-5
REM Usage: test_cuda_inference.bat [BUILD_DIR] [MODEL_FILE] [SAMPLES_DIR] [NUM_SAMPLES]

setlocal enabledelayedexpansion

REM Resolve defaults relative to script
set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%..\..\..\build\Debug"
set "MODEL_FILE=%SCRIPT_DIR%models\lenet5.onnx"
set "SAMPLES_DIR=%SCRIPT_DIR%test_samples"
set "NUM_SAMPLES=50"

REM Positional overrides
if not "%~1"=="" set "BUILD_DIR=%~1"
if not "%~2"=="" set "MODEL_FILE=%~2"
if not "%~3"=="" set "SAMPLES_DIR=%~3"
if not "%~4"=="" set "NUM_SAMPLES=%~4"

REM Locate executable (Debug then Release)
set "EXE_PATH=%BUILD_DIR%\bin\lenet5_cuda_inference.exe"
if not exist "%EXE_PATH%" (
    set "ALT_EXE=%SCRIPT_DIR%..\..\..\build\Release\bin\lenet5_cuda_inference.exe"
    if exist "!ALT_EXE!" (
        set "EXE_PATH=!ALT_EXE!"
    ) else (
        echo Error: lenet5_cuda_inference.exe not found.
        echo Checked:
        echo   %EXE_PATH%
        echo   !ALT_EXE!
        echo.
        echo Make sure you built with CUDA support enabled:
        echo   cmake -DMINI_INFER_ENABLE_CUDA=ON ...
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
echo LeNet-5 CUDA GPU Inference Test Script (Batch)
echo ======================================================================
echo.
echo Working directory: %cd%
for %%p in ("%EXE_PATH%") do set "BUILD_PARENT=%%~dpp"
for %%p in ("%BUILD_PARENT%") do set "BUILD_TYPE=%%~nxp"
echo Build type: %BUILD_TYPE%
echo.

echo Step 1: Running CUDA GPU Inference
echo ----------------------------------------------------------------------
echo Executable : %EXE_PATH%
echo Model      : %MODEL_FILE%
echo Samples    : %SAMPLES_DIR%
echo NumSamples : %NUM_SAMPLES%
echo.

if "%SAMPLES_BIN_FLAG%"=="1" (
    "%EXE_PATH%" "%MODEL_FILE%" "%SAMPLES_BIN%" %NUM_SAMPLES% > "%TEMP%\cuda_output.txt" 2>&1
) else (
    "%EXE_PATH%" "%MODEL_FILE%" "%SAMPLES_DIR%" %NUM_SAMPLES% > "%TEMP%\cuda_output.txt" 2>&1
)
set "code=%ERRORLEVEL%"
type "%TEMP%\cuda_output.txt"

if not "%code%"=="0" (
    echo.
    echo [FAIL] CUDA inference failed with exit code %code%.
    del "%TEMP%\cuda_output.txt" >nul 2>&1
    exit /b %code%
)

set "ACC="
set "TOTAL="
set "CORR="
for /f "tokens=2" %%a in ('findstr /C:"Accuracy:" "%TEMP%\cuda_output.txt"') do set "ACC=%%a"
for /f "tokens=3" %%a in ('findstr /C:"Total samples:" "%TEMP%\cuda_output.txt"') do set "TOTAL=%%a"
for /f "tokens=3" %%a in ('findstr /C:"Correct predictions:" "%TEMP%\cuda_output.txt"') do set "CORR=%%a"

set "PASS=0"
if defined ACC (
    for /f "tokens=1 delims=%%" %%a in ("!ACC!") do set "ACC_NUM=%%a"
    for /f "tokens=1 delims=." %%a in ("!ACC_NUM!") do set "ACC_INT=%%a"
    if !ACC_INT! GEQ 99 set "PASS=1"
)
if "!PASS!"=="0" if defined TOTAL if defined CORR (
    if "!TOTAL!"=="!CORR!" set "PASS=1"
)

echo.
echo ======================================================================
if "!PASS!"=="1" (
    echo [SUCCESS] CUDA GPU inference passed! Accuracy=!ACC! Total=!TOTAL! Correct=!CORR!
) else (
    echo [WARN] Inference succeeded but accuracy check failed or missing metrics.
    echo        Accuracy=!ACC! Total=!TOTAL! Correct=!CORR!
)
echo ======================================================================

del "%TEMP%\cuda_output.txt" >nul 2>&1

endlocal
