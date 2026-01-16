#!/bin/bash
#
# BERT Inference Test Script
# 对比 Mini-Infer 和 PyTorch 的 BERT 推理结果
#
# Usage:
#   ./test_bert.sh                    # 使用默认配置
#   ./test_bert.sh --verbose          # 显示详细对比
#   ./test_bert.sh --regenerate       # 重新生成 PyTorch 参考输出
#   ./test_bert.sh --build-type Release  # 使用 Release 构建
#

set -e

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 默认配置
BUILD_TYPE="Debug"
NUM_SAMPLES=10
VERBOSE=""
REGENERATE=""
CONDA_ENV="mini-infer"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --regenerate)
            REGENERATE="--regenerate-reference"
            shift
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --build-type TYPE    Build type: Debug or Release (default: Debug)"
            echo "  --num-samples N      Number of samples to test (default: 10)"
            echo "  --verbose, -v        Show detailed comparison"
            echo "  --regenerate         Regenerate PyTorch reference outputs"
            echo "  --conda-env ENV      Conda environment name (default: mini-infer)"
            echo "  --help, -h           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 路径配置
MINI_INFER_BIN="$PROJECT_ROOT/build/$BUILD_TYPE/bin/bert_inference"
MODEL_PATH="$SCRIPT_DIR/models/bert_tiny.onnx"
SAMPLES_DIR="$SCRIPT_DIR/test_samples"

echo "======================================================================"
echo "BERT Inference Test - Mini-Infer vs PyTorch"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  Project root: $PROJECT_ROOT"
echo "  Build type: $BUILD_TYPE"
echo "  Mini-Infer binary: $MINI_INFER_BIN"
echo "  Model: $MODEL_PATH"
echo "  Samples: $SAMPLES_DIR"
echo "  Num samples: $NUM_SAMPLES"
echo ""

# 检查 Mini-Infer 二进制文件
if [ ! -f "$MINI_INFER_BIN" ]; then
    echo "[ERROR] Mini-Infer binary not found: $MINI_INFER_BIN"
    echo ""
    echo "Please build the project first:"
    echo "  cmake --build build/$BUILD_TYPE --parallel"
    exit 1
fi

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] Model not found: $MODEL_PATH"
    echo ""
    echo "Please export the model first:"
    echo "  cd $SCRIPT_DIR && python export_bert.py"
    exit 1
fi

# 检查样本目录
if [ ! -d "$SAMPLES_DIR" ]; then
    echo "[ERROR] Samples directory not found: $SAMPLES_DIR"
    echo ""
    echo "Please generate samples first:"
    echo "  cd $SCRIPT_DIR && python export_bert_samples.py"
    exit 1
fi

# 激活 conda 环境（如果可用）
if command -v conda &> /dev/null; then
    echo "Activating conda environment: $CONDA_ENV"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" 2>/dev/null || {
        echo "[WARN] Could not activate conda environment: $CONDA_ENV"
        echo "       Continuing with current Python environment..."
    }
    echo ""
fi

# 运行对比测试
echo "Running comparison test..."
echo ""

python "$SCRIPT_DIR/test_bert_comparison.py" \
    --mini-infer-bin "$MINI_INFER_BIN" \
    --model "$MODEL_PATH" \
    --samples-dir "$SAMPLES_DIR" \
    --num-samples "$NUM_SAMPLES" \
    $VERBOSE \
    $REGENERATE

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] BERT inference test completed successfully!"
else
    echo "[FAILED] BERT inference test failed!"
fi

exit $EXIT_CODE
