@echo off
REM Batch script: benchmark optimized vs non-optimized inference
REM Usage:
REM   benchmark_optimization.bat [--build-dir DIR] [--model FILE] [--samples DIR] [--num-samples N] [--iterations K]

setlocal enabledelayedexpansion

REM Defaults
set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%..\..\..\build\Debug"
set "MODEL_FILE=%SCRIPT_DIR%models\lenet5.onnx"
set "SAMPLES_DIR=%SCRIPT_DIR%test_samples\binary"
set "NUM_SAMPLES=100"
set "ITERATIONS=20"
set "WARMUP=3"

REM Parse args (simple flag-value)
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--build-dir" (
    set "BUILD_DIR=%~2" & shift & shift & goto parse_args
)
if /i "%~1"=="--model" (
    set "MODEL_FILE=%~2" & shift & shift & goto parse_args
)
if /i "%~1"=="--samples" (
    set "SAMPLES_DIR=%~2" & shift & shift & goto parse_args
)
if /i "%~1"=="--num-samples" (
    set "NUM_SAMPLES=%~2" & shift & shift & goto parse_args
)
if /i "%~1"=="--iterations" (
    set "ITERATIONS=%~2" & shift & shift & goto parse_args
)
if /i "%~1"=="--warmup" (
    set "WARMUP=%~2" & shift & shift & goto parse_args
)
shift
goto parse_args
:args_done

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

echo ========================================
echo LeNet-5 Optimization Benchmark
echo ========================================
echo Executable : %EXE_PATH%
echo Model      : %MODEL_FILE%
echo Samples    : %SAMPLES_DIR%
echo NumSamples : %NUM_SAMPLES%
echo Iterations : %ITERATIONS%
echo Warmup     : %WARMUP%
echo.

REM Warmup (not measured)
if %WARMUP% GTR 0 (
    for /L %%w in (1,1,%WARMUP%) do (
        "%EXE_PATH%" "%MODEL_FILE%" "%SAMPLES_DIR%" %NUM_SAMPLES% --no-optimization >nul 2>&1
        "%EXE_PATH%" "%MODEL_FILE%" "%SAMPLES_DIR%" %NUM_SAMPLES% >nul 2>&1
    )
)

set "TEMP_ORIG=%TEMP%\bench_orig.txt"
set "TEMP_OPT=%TEMP%\bench_opt.txt"
if exist "%TEMP_ORIG%" del "%TEMP_ORIG%"
if exist "%TEMP_OPT%" del "%TEMP_OPT%"

for /L %%i in (1,1,%ITERATIONS%) do (
    echo Iteration %%i / %ITERATIONS%

    set "TIME1="
    "%EXE_PATH%" "%MODEL_FILE%" "%SAMPLES_DIR%" %NUM_SAMPLES% --no-optimization > "%TEMP%\out1.txt" 2>&1
    for /f "tokens=5" %%t in ('findstr /C:"Average time per sample:" "%TEMP%\out1.txt"') do (
        set "TIME1=%%t"
    )
    if defined TIME1 (
        echo   Original : !TIME1! ms
        echo !TIME1!>>"%TEMP_ORIG%"
    ) else (
        echo   Original : failed to parse timing
    )

    set "TIME2="
    "%EXE_PATH%" "%MODEL_FILE%" "%SAMPLES_DIR%" %NUM_SAMPLES% > "%TEMP%\out2.txt" 2>&1
    for /f "tokens=5" %%t in ('findstr /C:"Average time per sample:" "%TEMP%\out2.txt"') do (
        set "TIME2=%%t"
    )
    if defined TIME2 (
        echo   Optimized: !TIME2! ms
        echo !TIME2!>>"%TEMP_OPT%"
    ) else (
        echo   Optimized: failed to parse timing
    )
    echo.
)

if not exist "%TEMP_ORIG%" (
    echo No timing data collected. & exit /b 1
)
if not exist "%TEMP_OPT%" (
    echo No timing data collected. & exit /b 1
)

echo ========================================
echo Results (averaged)
echo ========================================
powershell -Command ^
  "$orig = Get-Content '%TEMP_ORIG%' | ForEach-Object {[double]$_};" ^
  "$opt  = Get-Content '%TEMP_OPT%'  | ForEach-Object {[double]$_};" ^
  "$a1=($orig|Measure-Object -Average).Average;" ^
  "$a2=($opt|Measure-Object -Average).Average;" ^
  "$spd=$a1/$a2;" ^
  "$imp=(($a1-$a2)/$a1)*100;" ^
  "Write-Host ('Original  : {0:F3} ms  ({1:F1} fps)' -f $a1, (1000/$a1));" ^
  "Write-Host ('Optimized : {0:F3} ms  ({1:F1} fps)' -f $a2, (1000/$a2));" ^
  "Write-Host ('Speedup   : {0:F2}x' -f $spd);" ^
  "Write-Host ('Time drop : {0:F1}%%' -f $imp);" ^
  "if($spd -gt 1.05){Write-Host 'PASS: Optimization is effective.' -ForegroundColor Green}" ^
  "elseif($spd -gt 0.95){Write-Host 'WARN: Optimization impact is minimal.' -ForegroundColor Yellow}" ^
  "else{Write-Host 'FAIL: Optimization degraded performance.' -ForegroundColor Red}"

if exist "%TEMP_ORIG%" del "%TEMP_ORIG%"
if exist "%TEMP_OPT%" del "%TEMP_OPT%"
if exist "%TEMP%\out1.txt" del "%TEMP%\out1.txt"
if exist "%TEMP%\out2.txt" del "%TEMP%\out2.txt"

endlocal

