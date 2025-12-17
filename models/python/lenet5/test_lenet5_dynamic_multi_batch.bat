@echo off
REM CMD script for exercising lenet5_dynamic_multi_batch example
REM Usage:
REM   test_lenet5_dynamic_multi_batch.bat [--max-batch N] [--accuracy-threshold X]
REM                                       [--model path] [--samples path] [--exe path]

setlocal enabledelayedexpansion

set MAX_BATCH=16
set ACCURACY_THRESHOLD=0.9
set MODEL_PATH=
set SAMPLES_DIR=
set EXECUTABLE=

:parse_args
if "%~1"=="" goto after_args
if /I "%~1"=="--max-batch" (
    set MAX_BATCH=%~2
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--accuracy-threshold" (
    set ACCURACY_THRESHOLD=%~2
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--model" (
    set MODEL_PATH=%~2
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--samples" (
    set SAMPLES_DIR=%~2
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--exe" (
    set EXECUTABLE=%~2
    shift
    shift
    goto parse_args
)
echo Unknown argument: %~1
exit /b 1

:after_args

echo ======================================================================
echo LeNet-5 Dynamic Multi-Batch Test (CMD)
echo ======================================================================
echo Working directory : %CD%
echo Requested max batch: %MAX_BATCH%
echo.

if not exist "lenet5_model.py" (
    echo [ERROR] Run this script from models\python\lenet5
    pause
    exit /b 1
)

if not exist "checkpoints\lenet5_best.pth" (
    echo [ERROR] Missing checkpoint checkpoints\lenet5_best.pth
    echo Train the model first: python train_lenet5.py --epochs 10
    pause
    exit /b 1
)

if "%MODEL_PATH%"=="" set "MODEL_PATH=%CD%\models\lenet5.onnx"
if "%SAMPLES_DIR%"=="" set "SAMPLES_DIR=%CD%\test_samples\dynamic_multi_batch"
set "BINARY_DIR=%SAMPLES_DIR%\binary"
set "CHECKPOINT=%CD%\checkpoints\lenet5_best.pth"

REM Determine batch sizes
set SAMPLES_REQUIRED=0
set BATCH_LIST=
for %%S in (1 4 8 12) do (
    if !MAX_BATCH! GEQ %%S (
        if "!ADDED_%%S!"=="" (
            set /a SAMPLES_REQUIRED+=%%S
            set "ADDED_%%S=1"
            if defined BATCH_LIST (
                set "BATCH_LIST=!BATCH_LIST!,%%S"
            ) else (
                set "BATCH_LIST=%%S"
            )
        )
    )
)
set EXTRA=%MAX_BATCH%
if %MAX_BATCH% GTR 16 set EXTRA=16
if %EXTRA% LEQ %MAX_BATCH% (
    if "!ADDED_%EXTRA%!"=="" (
        set /a SAMPLES_REQUIRED+=%EXTRA%
        set "ADDED_%EXTRA%=1"
        if defined BATCH_LIST (
            set "BATCH_LIST=!BATCH_LIST!,%EXTRA%"
        ) else (
            set "BATCH_LIST=%EXTRA%"
        )
    )
)
if %SAMPLES_REQUIRED% LSS 1 (
    set SAMPLES_REQUIRED=%MAX_BATCH%
    set "BATCH_LIST=%MAX_BATCH%"
)

echo Target batch sizes  : !BATCH_LIST!
echo Required sample count: %SAMPLES_REQUIRED%
echo.

if not exist "%MODEL_PATH%" (
    echo Step 1: Exporting ONNX model
    echo ----------------------------------------------------------------------
    python export_lenet5.py ^
        --checkpoint "%CHECKPOINT%" ^
        --format onnx ^
        --output "%MODEL_PATH%"
    if errorlevel 1 (
        echo [ERROR] Failed to export ONNX model.
        pause
        exit /b 1
    )
    echo.
)

REM Count labeled samples
set LABEL_COUNT=0
if exist "%BINARY_DIR%" (
    for /f %%C in ('dir /b "%BINARY_DIR%\*_label_*.bin" 2^>nul ^| find /c /v ""') do set LABEL_COUNT=%%C
) else (
    set LABEL_COUNT=0
)

if %LABEL_COUNT% LSS %SAMPLES_REQUIRED% (
    set /a NUM_PER_CLASS=(SAMPLES_REQUIRED + 9) / 10
    if %NUM_PER_CLASS% LSS 5 set NUM_PER_CLASS=5
    echo Step 2: Exporting MNIST samples (num-per-class=%NUM_PER_CLASS%)
    echo ----------------------------------------------------------------------
    python export_mnist_samples.py ^
        --output-dir "%SAMPLES_DIR%" ^
        --num-per-class %NUM_PER_CLASS% ^
        --formats binary png
    if errorlevel 1 (
        echo [ERROR] Failed to export MNIST samples.
        pause
        exit /b 1
    )
    echo.
    if exist "%BINARY_DIR%" (
        for /f %%C in ('dir /b "%BINARY_DIR%\*_label_*.bin" 2^>nul ^| find /c /v ""') do set LABEL_COUNT=%%C
    ) else (
        set LABEL_COUNT=0
    )
)

if %LABEL_COUNT% LSS %SAMPLES_REQUIRED% (
    echo [ERROR] Only %LABEL_COUNT% labeled samples available under %BINARY_DIR%.
    pause
    exit /b 1
)

echo Sample directory    : %BINARY_DIR%
echo Available samples   : %LABEL_COUNT%
echo.

if not defined EXECUTABLE (
    if exist "..\..\..\build\windows-debug\bin\lenet5_dynamic_multi_batch.exe" (
        set "EXECUTABLE=..\..\..\build\windows-debug\bin\lenet5_dynamic_multi_batch.exe"
    ) else if exist "..\..\..\build\Debug\bin\lenet5_dynamic_multi_batch.exe" (
        set "EXECUTABLE=..\..\..\build\Debug\bin\lenet5_dynamic_multi_batch.exe"
    ) else if exist "..\..\..\build\windows-release\bin\lenet5_dynamic_multi_batch.exe" (
        set "EXECUTABLE=..\..\..\build\windows-release\bin\lenet5_dynamic_multi_batch.exe"
    ) else if exist "..\..\..\build\Release\bin\lenet5_dynamic_multi_batch.exe" (
        set "EXECUTABLE=..\..\..\build\Release\bin\lenet5_dynamic_multi_batch.exe"
    )
)

if not defined EXECUTABLE (
    echo [ERROR] lenet5_dynamic_multi_batch executable not found.
    echo Build it first: cmake --build build --config Debug --target lenet5_dynamic_multi_batch
    pause
    exit /b 1
)

echo Using executable    : %EXECUTABLE%
echo Accuracy threshold  : %ACCURACY_THRESHOLD%
echo.

echo Step 3: Running lenet5_dynamic_multi_batch
echo ----------------------------------------------------------------------
"%EXECUTABLE%" ^
    --model "%MODEL_PATH%" ^
    --samples "%BINARY_DIR%" ^
    --accuracy-threshold %ACCURACY_THRESHOLD% ^
    --max-batch %MAX_BATCH%
set EXIT_CODE=%errorlevel%
echo.

echo ======================================================================
if %EXIT_CODE%==0 (
    echo [SUCCESS] Dynamic multi-batch test completed.
) else (
    echo [FAILED] Dynamic multi-batch test failed with exit code %EXIT_CODE%.
)
echo ======================================================================
echo Model path   : %MODEL_PATH%
echo Samples used : %BINARY_DIR%
echo Batch sizes  : !BATCH_LIST!
echo.

pause
exit /b %EXIT_CODE%
