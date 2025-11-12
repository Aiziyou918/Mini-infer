# Windows PowerShell 构建脚本

param(
    [Parameter()]
    [ValidateSet("Debug", "Release")]
    [string]$BuildType = "Release",
    
    [Parameter()]
    [switch]$Clean,
    
    [Parameter()]
    [switch]$Test,
    
    [Parameter()]
    [switch]$Install
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Mini-Infer 构建脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 项目根目录
$ProjectRoot = $PSScriptRoot
$BuildDir = Join-Path $ProjectRoot "build"

# 清理构建目录
if ($Clean) {
    Write-Host "清理构建目录..." -ForegroundColor Yellow
    if (Test-Path $BuildDir) {
        Remove-Item -Recurse -Force $BuildDir
    }
    Write-Host "清理完成" -ForegroundColor Green
}

# 创建构建目录
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# 切换到构建目录
Set-Location $BuildDir

# 配置 CMake
Write-Host "`n配置 CMake ($BuildType)..." -ForegroundColor Yellow
cmake .. -DCMAKE_BUILD_TYPE=$BuildType `
         -DMINI_INFER_BUILD_TESTS=ON `
         -DMINI_INFER_BUILD_EXAMPLES=ON

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake 配置失败" -ForegroundColor Red
    exit 1
}

# 构建项目
Write-Host "`n开始构建..." -ForegroundColor Yellow
cmake --build . --config $BuildType --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Host "构建失败" -ForegroundColor Red
    exit 1
}

Write-Host "`n构建成功!" -ForegroundColor Green

# 运行测试
if ($Test) {
    Write-Host "`n运行测试..." -ForegroundColor Yellow
    ctest --output-on-failure -C $BuildType
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "所有测试通过!" -ForegroundColor Green
    } else {
        Write-Host "测试失败" -ForegroundColor Red
        exit 1
    }
}

# 安装
if ($Install) {
    Write-Host "`n安装..." -ForegroundColor Yellow
    cmake --install . --config $BuildType
    Write-Host "安装完成" -ForegroundColor Green
}

# 返回项目根目录
Set-Location $ProjectRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "完成!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "可执行文件位置: $BuildDir\bin\$BuildType" -ForegroundColor Yellow
Write-Host "库文件位置: $BuildDir\lib\$BuildType" -ForegroundColor Yellow

