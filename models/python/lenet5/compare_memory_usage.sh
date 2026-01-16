#!/bin/bash
# ============================================================================
# Memory Usage Comparison Wrapper Script (Bash)
# ============================================================================

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Memory Usage Comparison${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}[ERROR] Python not found. Please install Python 3.x${NC}"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}Using: $PYTHON_VERSION${NC}"
echo ""

# File paths
OPTIMIZED_FILE="test_samples/optimized_no_memory_outputs.json"
OPTIMIZED_MEMORY_FILE="test_samples/optimized_memory_outputs.json"

# Check if result files exist
FILES_EXIST=0
if [ -f "$OPTIMIZED_FILE" ]; then
    FILES_EXIST=$((FILES_EXIST + 1))
fi
if [ -f "$OPTIMIZED_MEMORY_FILE" ]; then
    FILES_EXIST=$((FILES_EXIST + 1))
fi

if [ $FILES_EXIST -eq 0 ]; then
    echo -e "${YELLOW}[WARNING] No result files found${NC}"
    echo -e "${YELLOW}Please run ./test_optimized_with_memory.sh first${NC}"
    echo ""
    echo "Expected files:"
    echo "  - $OPTIMIZED_FILE"
    echo "  - $OPTIMIZED_MEMORY_FILE"
    echo ""
    exit 1
fi

echo -e "${GREEN}Found $FILES_EXIST result file(s)${NC}"
echo ""

# Run comparison script
$PYTHON_CMD compare_memory_usage.py "$OPTIMIZED_FILE" "$OPTIMIZED_MEMORY_FILE"

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}[ERROR] Comparison failed${NC}"
    exit 1
fi

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${GREEN}Comparison completed successfully!${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
