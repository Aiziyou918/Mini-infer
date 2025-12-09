@echo off
REM ============================================================================
REM Memory Usage Comparison Wrapper Script
REM ============================================================================

setlocal enabledelayedexpansion

echo ========================================
echo Memory Usage Comparison
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if "%ERRORLEVEL%" neq "0" (
    echo [ERROR] Python not found. Please install Python 3.x
    exit /b 1
)

REM File paths
set OPTIMIZED_FILE=test_samples\optimized_no_memory_outputs.json
set OPTIMIZED_MEMORY_FILE=test_samples\optimized_memory_outputs.json

REM Check if result files exist
set FILES_EXIST=0
if exist "%OPTIMIZED_FILE%" set /a FILES_EXIST+=1
if exist "%OPTIMIZED_MEMORY_FILE%" set /a FILES_EXIST+=1

if "%FILES_EXIST%"=="0" (
    echo [WARNING] No result files found
    echo Please run test_optimized_with_memory.bat first
    echo.
    echo Expected files:
    echo   - %OPTIMIZED_FILE%
    echo   - %OPTIMIZED_MEMORY_FILE%
    echo.
    pause
    exit /b 1
)

echo Found %FILES_EXIST% result file(s)
echo.

REM Run comparison script
python compare_memory_usage.py "%OPTIMIZED_FILE%" "%OPTIMIZED_MEMORY_FILE%"

if "%ERRORLEVEL%" neq "0" (
    echo.
    echo [ERROR] Comparison failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Comparison completed successfully!
echo ========================================
echo.

pause
endlocal
