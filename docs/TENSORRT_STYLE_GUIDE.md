# TensorRT风格Kernel架构指南

## 🎯 架构概览

Mini-Infer现在采用TensorRT风格的Kernel管理架构：

```
┌─────────────────────────────────┐
│  算子层 (Operators)              │
│  Conv2D, Linear, etc.           │
└───────────┬─────────────────────┘
            │
┌───────────▼─────────────────────┐
│  Kernel接口层                    │
│  GEMMKernel, Im2ColKernel       │
└───────────┬─────────────────────┘
            │
┌───────────▼─────────────────────┐
│  Registry系统 (自动dispatch)     │
│  - GEMMRegistry_NN              │
│  - GEMMRegistry_NT              │
│  - Im2ColRegistry               │
└───────────┬─────────────────────┘
            │
    ┌───────┴────────┐
┌───▼───┐      ┌─────▼──────┐
│  CPU  │      │   CUDA     │
│ Impl  │      │   Impl     │
│(自注册)│      │  (自注册)   │
└───────┘      └────────────┘
```

## ✨ 核心特性

### 1. **自动注册（Auto-Registration）**

类似TensorRT的IPluginRegistry，实现在程序启动时自动注册：

```cpp
// src/kernels/cpu/gemm_cpu.cpp

namespace cpu {
    // 实现
    template<typename T>
    void gemm_nn_impl(const T* A, const T* B, T* C, int M, int N, int K) {
        // CPU实现
    }
    
    // 可用性检查
    bool is_cpu_available() {
        return true;
    }
}

// 自动注册（程序启动时执行）
static auto register_gemm_nn_float = AutoRegister<
    GEMMRegistry_NN<float>,      // 注册表类型
    GEMMFunc_NN<float>            // 函数类型
>(
    KernelBackend::CPU,           // Backend类型
    cpu::gemm_nn_impl<float>,     // 函数指针
    cpu::is_cpu_available,        // 可用性检查器
    100                           // 优先级（越高越优先）
);
```

### 2. **Registry Dispatch**

运行时自动选择最优实现：

```cpp
template<typename T>
void GEMMKernel::gemm_nn(..., KernelBackend backend) {
    // 从Registry获取kernel函数
    auto func = GEMMRegistry_NN<T>::instance().get_best_kernel();
    
    // 执行
    func(A, B, C, M, N, K);
}
```

### 3. **零虚函数开销**

使用函数指针代替虚函数：
- ✅ 无vtable查找
- ✅ 支持inline优化
- ✅ 兼容CUDA kernel

## 📝 添加新Backend实现

### 示例：添加AVX2优化版本

#### Step 1: 创建实现文件

```cpp
// src/kernels/cpu/gemm_cpu_avx2.cpp

#include "mini_infer/kernels/gemm.h"
#include <immintrin.h>  // AVX2 intrinsics

namespace mini_infer {
namespace kernels {
namespace cpu {
namespace avx2 {

// AVX2优化实现
template<typename T>
void gemm_nn_impl(const T* A, const T* B, T* C, int M, int N, int K) {
    // AVX2向量化实现
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; n += 8) {  // Process 8 floats at once
            __m256 sum = _mm256_setzero_ps();
            
            for (int k = 0; k < K; ++k) {
                __m256 a = _mm256_broadcast_ss(&A[m * K + k]);
                __m256 b = _mm256_load_ps(&B[k * N + n]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }
            
            _mm256_store_ps(&C[m * N + n], sum);
        }
    }
}

// 检查AVX2支持
bool is_avx2_available() {
    #ifdef __AVX2__
        return true;
    #else
        // Runtime detection
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2");
    #endif
}

} // namespace avx2
} // namespace cpu

// 自动注册AVX2版本
static auto register_gemm_nn_float_avx2 = AutoRegister<
    GEMMRegistry_NN<float>,
    GEMMFunc_NN<float>
>(
    KernelBackend::CPU_AVX2,
    cpu::avx2::gemm_nn_impl<float>,
    cpu::avx2::is_avx2_available,
    200  // 更高优先级（优先使用AVX2）
);

} // namespace kernels
} // namespace mini_infer
```

#### Step 2: 更新CMakeLists.txt

```cmake
# src/kernels/CMakeLists.txt

set(KERNEL_SOURCES
    cpu/gemm_cpu.cpp
    cpu/gemm_cpu_avx2.cpp  # 新增
    cpu/im2col_cpu.cpp
)

# AVX2编译选项
if(MSVC)
    set_source_files_properties(cpu/gemm_cpu_avx2.cpp 
        PROPERTIES COMPILE_FLAGS "/arch:AVX2")
else()
    set_source_files_properties(cpu/gemm_cpu_avx2.cpp 
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
endif()
```

#### Step 3: 自动生效

无需修改任何其他代码！Registry会自动：
1. 启动时注册AVX2版本
2. 检测CPU是否支持AVX2
3. 如果支持，优先使用AVX2版本

## 🚀 添加CUDA Backend

### 示例：CUDA实现

```cpp
// src/kernels/cuda/gemm_cuda.cu

#include "mini_infer/kernels/gemm.h"

namespace mini_infer {
namespace kernels {
namespace cuda {

// CUDA kernel
template<typename T>
__global__ void gemm_nn_kernel(const T* A, const T* B, T* C, 
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host wrapper
template<typename T>
void gemm_nn_impl(const T* A, const T* B, T* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    gemm_nn_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

// CUDA可用性检查
bool is_cuda_available() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count > 0;
}

} // namespace cuda

// 自动注册CUDA版本
static auto register_gemm_nn_float_cuda = AutoRegister<
    GEMMRegistry_NN<float>,
    GEMMFunc_NN<float>
>(
    KernelBackend::CUDA,
    cuda::gemm_nn_impl<float>,
    cuda::is_cuda_available,
    300  // 最高优先级（优先使用CUDA）
);

} // namespace kernels
} // namespace mini_infer
```

## 📊 优先级系统

Registry按优先级降序选择kernel：

| Backend | 优先级 | 说明 |
|---------|--------|------|
| CUDA_CUBLAS | 400 | cuBLAS优化 |
| CUDA | 300 | 基础CUDA |
| CPU_BLAS | 250 | OpenBLAS/MKL |
| CPU_AVX512 | 220 | AVX512向量化 |
| CPU_AVX2 | 200 | AVX2向量化 |
| CPU | 100 | 基础CPU实现 |

## 🎯 与TensorRT对比

### TensorRT Plugin系统

```cpp
// TensorRT风格
class MyPlugin : public IPluginV2 {
    int enqueue(...) override {
        // 调用kernel
        myKernel<<<>>>(...);
    }
};

// 注册Plugin
REGISTER_TENSORRT_PLUGIN(MyPluginCreator);
```

### Mini-Infer Kernel系统

```cpp
// Mini-Infer风格
namespace cpu {
    void my_kernel(...) { /* 实现 */ }
}

// 自动注册
static auto reg = AutoRegister<MyRegistry, MyFunc>(
    KernelBackend::CPU,
    cpu::my_kernel,
    []() { return true; },
    100
);
```

**共同点**：
- ✅ 自动注册机制
- ✅ 运行时dispatch
- ✅ 零虚函数开销
- ✅ 支持多Backend

**差异**：
- TensorRT: Plugin是算子层，使用虚函数
- Mini-Infer: Kernel是计算层，使用函数指针

## 🔄 迁移指南

### 从旧风格迁移

**旧代码（手动dispatch）**：
```cpp
template<typename T>
void GEMMKernel::gemm_nn(..., KernelBackend backend) {
    switch(backend) {
        case CPU:
            cpu::gemm_nn_impl(...);
            break;
        case CUDA:
            cuda::gemm_nn_impl(...);
            break;
    }
}
```

**新代码（自动dispatch）**：
```cpp
template<typename T>
void GEMMKernel::gemm_nn(..., KernelBackend backend) {
    auto func = GEMMRegistry_NN<T>::instance().get_best_kernel();
    func(...);
}
```

### 向后兼容

公共接口保持不变：
```cpp
// 用户代码无需修改
GEMMKernel::gemm_nn<float>(A, B, C, M, N, K);
```

## 📚 参考资料

- [TensorRT Plugin Development](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#add_custom_layer)
- [TensorRT IPluginRegistry](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_registry.html)
- [PyTorch Dispatcher](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/dispatch/Dispatcher.h)

## ✅ 最佳实践

1. **一个Backend一个文件** - 易于管理和编译
2. **使用命名空间** - `cpu::`, `cuda::`, `avx2::` 等
3. **提供可用性检查** - 运行时检测硬件支持
4. **设置合理优先级** - 确保自动选择最优实现
5. **保持接口简单** - 函数签名尽量简洁
6. **添加文档注释** - 说明实现特性和优化

## 🎉 总结

TensorRT风格的Kernel架构带来：
- ✅ 零开销抽象
- ✅ 自动Backend选择
- ✅ 易于扩展
- ✅ 代码解耦
- ✅ 工业级设计

完美对齐TensorRT的设计理念！🚀
