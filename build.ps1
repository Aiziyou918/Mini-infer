#!/usr/bin/env pwsh
# Mini-Infer Conan 自动化构建脚本 (Windows)

param(
    [ValidateSet("Debug", "Release")]
    [string]$BuildType = "Debug",
    
    [switch]$EnableOnnx = $true,
    [switch]$EnableLogging = $true,
    [switch]$EnableCuda = $false,
    [switch]$Clean,
    [switch]$Test,
    [switch]$Install
)

$ErrorActionPreference = "Stop"

# 颜色输出
function Write-Info { param([string]$Message) Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Success { param([string]$Message) Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
function Write-Error-Msg { param([string]$Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }

Write-Info "========================================="
Write-Info "Mini-Infer 自动化构建脚本 (Conan)"
Write-Info "========================================="
Write-Info "构建类型: $BuildType"
Write-Info "ONNX支持: $EnableOnnx"
Write-Info "日志支持: $EnableLogging"
Write-Info "CUDA支持: $EnableCuda"
Write-Info ""

# 项目配置
$ProjectRoot = $PSScriptRoot
$BuildDir = Join-Path $ProjectRoot "build" $BuildType
$PresetName = "conan-$($BuildType.ToLower())"

try {
    # 步骤 1: 清理（如果需要）
    if ($Clean) {
        Write-Info "清理构建目录..."
        if (Test-Path $BuildDir) {
            Remove-Item -Recurse -Force $BuildDir
        }
        Write-Success "清理完成"
    }

    # 步骤 2: 检查 Conan
    Write-Info "检查 Conan 安装..."
    $conanVersion = & conan --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Msg "Conan 未安装! 请运行: pip install conan"
        exit 1
    }
    Write-Success "找到 Conan: $conanVersion"

    # 步骤 3: 检查 Ninja
    $UseNinja = $false
    Write-Info "检查 Ninja 生成器..."
    $ninjaCheck = & ninja --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "找到 Ninja: $ninjaCheck"
        $UseNinja = $true
    } else {
        Write-Info "未检测到 Ninja 生成器"
        Write-Host ""
        Write-Host "Ninja 可以显著提升编译速度（提升 50%+）" -ForegroundColor Yellow
        Write-Host "是否要安装 Ninja？[Y/n]: " -NoNewline -ForegroundColor Yellow
        $choice = Read-Host
        
        if ($choice -eq "" -or $choice -eq "Y" -or $choice -eq "y") {
            Write-Info "尝试安装 Ninja..."
            Write-Host ""
            Write-Host "请选择安装方式:" -ForegroundColor Cyan
            Write-Host "  1. scoop install ninja     (推荐)" -ForegroundColor White
            Write-Host "  2. choco install ninja" -ForegroundColor White
            Write-Host "  3. pip install ninja" -ForegroundColor White
            Write-Host ""
            Write-Host "请在另一个终端执行安装命令，完成后按回车继续..." -ForegroundColor Yellow
            Read-Host
            
            # 重新检查
            $ninjaCheck = & ninja --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Ninja 安装成功: $ninjaCheck"
                $UseNinja = $true
            } else {
                Write-Info "未检测到 Ninja，将使用默认生成器 (Visual Studio)"
            }
        } else {
            Write-Info "跳过 Ninja 安装，使用默认生成器 (Visual Studio)"
        }
    }

    # 步骤 4: 安装依赖
    Write-Info "安装依赖..."
    $conanArgs = @(
        "install", ".",
        "-s", "build_type=$BuildType",
        "-o", "enable_onnx=$($EnableOnnx.ToString().ToLower())",
        "-o", "enable_logging=$($EnableLogging.ToString().ToLower())",
        "-o", "enable_cuda=$($EnableCuda.ToString().ToLower())",
        "--build=missing"
    )
    
    # 如果使用 Ninja，添加生成器配置
    if ($UseNinja) {
        $conanArgs += @("-c", "tools.cmake.cmaketoolchain:generator=Ninja")
        Write-Info "使用 Ninja 生成器"
    } else {
        Write-Info "使用 Visual Studio 生成器"
    }
    
    & conan @conanArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Msg "Conan install 失败!"
        exit 1
    }
    Write-Success "依赖安装完成"

    # 步骤 5: 配置 CMake
    Write-Info "配置 CMake..."
    cmake --preset $PresetName
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Msg "CMake 配置失败!"
        exit 1
    }
    Write-Success "CMake 配置完成"

    # 步骤 6: 编译
    Write-Info "编译项目..."
    cmake --build $BuildDir --config $BuildType
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Msg "编译失败!"
        exit 1
    }
    Write-Success "编译完成"

    # 步骤 7: 测试（可选）
    if ($Test) {
        Write-Info "运行测试..."
        ctest --preset $PresetName --output-on-failure
        if ($LASTEXITCODE -eq 0) {
            Write-Success "所有测试通过!"
        } else {
            Write-Error-Msg "测试失败!"
            exit 1
        }
    }

    # 步骤 8: 安装（可选）
    if ($Install) {
        Write-Info "安装到 install 目录..."
        cmake --install $BuildDir --prefix install
        if ($LASTEXITCODE -eq 0) {
            Write-Success "安装完成"
        } else {
            Write-Error-Msg "安装失败!"
            exit 1
        }
    }

    # 完成
    Write-Info ""
    Write-Success "========================================="
    Write-Success "构建成功完成!"
    Write-Success "========================================="
    Write-Info ""
    Write-Info "二进制文件: $BuildDir\bin\"
    if ($EnableOnnx) {
        Write-Info ""
        Write-Info "运行示例:"
        Write-Host "  .\$BuildDir\bin\onnx_parser_example.exe .\models\python\lenet5\models\lenet5.onnx" -ForegroundColor White
    }

} catch {
    Write-Error-Msg "构建过程失败: $_"
    exit 1
}

