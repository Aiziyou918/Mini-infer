#!/bin/bash
# Mini-Infer Conan 自动化构建脚本 (Linux/macOS)

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 默认参数
BUILD_TYPE="Debug"
ENABLE_ONNX=true
ENABLE_LOGGING=true
ENABLE_CUDA=false
CLEAN=false
RUN_TESTS=false
INSTALL=false

# 辅助函数
info() { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 帮助信息
show_help() {
    cat << EOF
用法: $0 [选项]

选项:
    -d, --debug              Debug 构建 (默认)
    -r, --release            Release 构建
    --no-onnx                禁用 ONNX 支持
    --no-logging             禁用日志支持
    --enable-cuda            启用 CUDA 支持
    -c, --clean              清理构建目录
    -t, --test               运行测试
    -i, --install            安装到 install 目录
    -h, --help               显示帮助

示例:
    $0                       # Debug 构建，启用 ONNX
    $0 -r                    # Release 构建
    $0 -r --no-onnx -c       # Release，禁用 ONNX，清理构建
    $0 -t -i                 # 测试并安装
EOF
}

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
        --no-onnx)
            ENABLE_ONNX=false
            shift
            ;;
        --no-logging)
            ENABLE_LOGGING=false
            shift
            ;;
        --enable-cuda)
            ENABLE_CUDA=true
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
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 项目配置
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build/$BUILD_TYPE"
PRESET_NAME="conan-$(echo $BUILD_TYPE | tr '[:upper:]' '[:lower:]')"

info "========================================="
info "Mini-Infer 自动化构建脚本 (Conan)"
info "========================================="
info "构建类型: $BUILD_TYPE"
info "ONNX支持: $ENABLE_ONNX"
info "日志支持: $ENABLE_LOGGING"
info "CUDA支持: $ENABLE_CUDA"
echo ""

# 步骤 1: 清理（如果需要）
if [ "$CLEAN" = true ]; then
    info "清理构建目录..."
    rm -rf "$BUILD_DIR"
    success "清理完成"
fi

# 步骤 2: 检查 Conan
info "检查 Conan 安装..."
if ! command -v conan &> /dev/null; then
    error "Conan 未安装! 请运行: pip install conan"
    exit 1
fi
CONAN_VERSION=$(conan --version)
success "找到 Conan: $CONAN_VERSION"

# 步骤 3: 检查 Ninja
USE_NINJA=false
info "检查 Ninja 生成器..."
if command -v ninja &> /dev/null; then
    NINJA_VERSION=$(ninja --version)
    success "找到 Ninja: $NINJA_VERSION"
    USE_NINJA=true
else
    info "未检测到 Ninja 生成器"
    echo ""
    echo -e "${YELLOW}Ninja 可以显著提升编译速度（提升 50%+）${NC}"
    echo -n -e "${YELLOW}是否要安装 Ninja？[Y/n]: ${NC}"
    read -r choice
    
    if [[ "$choice" == "" || "$choice" == "Y" || "$choice" == "y" ]]; then
        info "尝试安装 Ninja..."
        echo ""
        echo -e "${CYAN}请选择安装方式:${NC}"
        echo "  1. Ubuntu/Debian: sudo apt-get install ninja-build"
        echo "  2. CentOS/RHEL:   sudo yum install ninja-build"
        echo "  3. macOS:         brew install ninja"
        echo "  4. 通用:          pip install ninja"
        echo ""
        echo -e "${YELLOW}请在另一个终端执行安装命令，完成后按回车继续...${NC}"
        read -r
        
        # 重新检查
        if command -v ninja &> /dev/null; then
            NINJA_VERSION=$(ninja --version)
            success "Ninja 安装成功: $NINJA_VERSION"
            USE_NINJA=true
        else
            info "未检测到 Ninja，将使用默认生成器 (Unix Makefiles)"
        fi
    else
        info "跳过 Ninja 安装，使用默认生成器 (Unix Makefiles)"
    fi
fi

# 步骤 4: 安装依赖
info "安装依赖..."

# 构建 conan install 命令参数
CONAN_ARGS=(
    "install" "."
    "-s" "build_type=$BUILD_TYPE"
    "-o" "enable_onnx=$ENABLE_ONNX"
    "-o" "enable_logging=$ENABLE_LOGGING"
    "-o" "enable_cuda=$ENABLE_CUDA"
    "--build=missing"
)

# 如果使用 Ninja，添加生成器配置
if [ "$USE_NINJA" = true ]; then
    CONAN_ARGS+=("-c" "tools.cmake.cmaketoolchain:generator=Ninja")
    info "使用 Ninja 生成器"
else
    info "使用 Unix Makefiles 生成器"
fi

conan "${CONAN_ARGS[@]}"

if [ $? -ne 0 ]; then
    error "Conan install 失败!"
    exit 1
fi
success "依赖安装完成"

# 步骤 5: 配置 CMake
info "配置 CMake..."
cmake --preset $PRESET_NAME

if [ $? -ne 0 ]; then
    error "CMake 配置失败!"
    exit 1
fi
success "CMake 配置完成"

# 步骤 6: 编译
info "编译项目..."
cmake --build $BUILD_DIR

if [ $? -ne 0 ]; then
    error "编译失败!"
    exit 1
fi
success "编译完成"

# 步骤 7: 测试（可选）
if [ "$RUN_TESTS" = true ]; then
    info "运行测试..."
    ctest --preset $PRESET_NAME --output-on-failure
    
    if [ $? -eq 0 ]; then
        success "所有测试通过!"
    else
        error "测试失败!"
        exit 1
    fi
fi

# 步骤 8: 安装（可选）
if [ "$INSTALL" = true ]; then
    info "安装到 install 目录..."
    cmake --install $BUILD_DIR --prefix install
    
    if [ $? -eq 0 ]; then
        success "安装完成"
    else
        error "安装失败!"
        exit 1
    fi
fi

# 完成
echo ""
success "========================================="
success "构建成功完成!"
success "========================================="
echo ""
info "二进制文件: $BUILD_DIR/bin/"
if [ "$ENABLE_ONNX" = true ]; then
    echo ""
    info "运行示例:"
    echo "  $BUILD_DIR/bin/onnx_parser_example ./models/python/lenet5/models/lenet5.onnx"
fi

