# ============================================================================
# Memory Usage Comparison Wrapper Script (PowerShell)
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Memory Usage Comparison" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Using: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found. Please install Python 3.x" -ForegroundColor Red
    exit 1
}

# File paths
$OPTIMIZED_FILE = "test_samples\optimized_no_memory_outputs.json"
$OPTIMIZED_MEMORY_FILE = "test_samples\optimized_memory_outputs.json"

# Check if result files exist
$filesExist = 0
if (Test-Path $OPTIMIZED_FILE) { $filesExist++ }
if (Test-Path $OPTIMIZED_MEMORY_FILE) { $filesExist++ }

if ($filesExist -eq 0) {
    Write-Host "[WARNING] No result files found" -ForegroundColor Yellow
    Write-Host "Please run test_optimized_with_memory.ps1 first" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Expected files:"
    Write-Host "  - $OPTIMIZED_FILE"
    Write-Host "  - $OPTIMIZED_MEMORY_FILE"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Found $filesExist result file(s)" -ForegroundColor Green
Write-Host ""

# Run comparison script
python compare_memory_usage.py $OPTIMIZED_FILE $OPTIMIZED_MEMORY_FILE

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Comparison failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Comparison completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit"
