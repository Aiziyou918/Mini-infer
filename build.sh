#!/bin/bash
# Linux/macOS 构建脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 默认参数
BUILD_TYPE="Release"
CLEAN=false
RUN_TESTS=false
INSTALL=false
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -r|--release)
            BUILD_TYPE="Release"
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -t|--test)
            RUN_TESTS=true
            shift
            ;;
        -i|--install)
            INSTALL=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  -d, --debug      Debug 构建"
            echo "  -r, --release    Release 构建 (默认)"
            echo "  -c, --clean      清理构建目录"
            echo "  -t, --test       运行测试"
            echo "  -i, --install    安装"
            echo "  -j, --jobs N     并行构建任务数 (默认: $JOBS)"
            echo "  -h, --help       显示帮助"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}========================================"
echo "Mini-Infer 构建脚本"
echo -e "========================================${NC}"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

# 清理构建目录
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}清理构建目录...${NC}"
    rm -rf "$BUILD_DIR"
    echo -e "${GREEN}清理完成${NC}"
fi

# 创建构建目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 配置 CMake
echo -e "\n${YELLOW}配置 CMake ($BUILD_TYPE)...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DMINI_INFER_BUILD_TESTS=ON \
    -DMINI_INFER_BUILD_EXAMPLES=ON

# 构建项目
echo -e "\n${YELLOW}开始构建...${NC}"
cmake --build . --parallel $JOBS

echo -e "\n${GREEN}构建成功!${NC}"

# 运行测试
if [ "$RUN_TESTS" = true ]; then
    echo -e "\n${YELLOW}运行测试...${NC}"
    ctest --output-on-failure
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}所有测试通过!${NC}"
    else
        echo -e "${RED}测试失败${NC}"
        exit 1
    fi
fi

# 安装
if [ "$INSTALL" = true ]; then
    echo -e "\n${YELLOW}安装...${NC}"
    cmake --install .
    echo -e "${GREEN}安装完成${NC}"
fi

# 返回项目根目录
cd "$PROJECT_ROOT"

echo -e "\n${CYAN}========================================"
echo "完成!"
echo -e "========================================${NC}"
echo ""
echo -e "${YELLOW}可执行文件位置: $BUILD_DIR/bin${NC}"
echo -e "${YELLOW}库文件位置: $BUILD_DIR/lib${NC}"

