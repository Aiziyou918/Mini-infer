# Mini-Infer Kernel Layer

## 📖 概述

Kernel层是Mini-Infer的计算核心，提供高性能的算子实现。设计参考TensorRT的Plugin架构，支持多Backend（CPU/GPU）和灵活扩展。

## 🏗️ 架构设计

```
┌─────────────────────────────────────┐
│        Operator Layer               │
│   (Conv2D, Linear, ReLU, ...)       │
└───────────────┬─────────────────────┘
                │
┌───────────────▼─────────────────────┐
│        Kernel Interface             │
│   (GEMMKernel, Im2ColKernel, ...)   │
└───────────────┬─────────────────────┘
                │
        ┌───────┴────────┐
        │                │
┌───────▼──────┐  ┌──────▼────────┐
│ CPU Kernels  │  │  CUDA Kernels │
│              │  │   (未来)       │
│ - gemm_cpu   │  │ - gemm_cuda   │
│ - im2col_cpu │  │ - im2col_cuda │
└──────────────┘  └───────────────┘
```

## 📁 目录结构

```
kernels/
├── README.md           # 本文件
├── CMakeLists.txt      # 构建配置
└── cpu/                # CPU实现
    ├── gemm_cpu.cpp    # GEMM实现
    └── im2col_cpu.cpp  # Im2Col实现
```

### 未来规划
```
kernels/
├── cuda/               # GPU实现
│   ├── gemm_cuda.cu
│   ├── gemm_cublas.cu
│   └── im2col_cuda.cu
├── cpu/
│   ├── gemm_cpu.cpp          # 基础实现
│   ├── gemm_cpu_avx2.cpp     # AVX2优化
│   ├── gemm_cpu_avx512.cpp   # AVX512优化
│   └── gemm_blas.cpp         # BLAS包装
└── arm/                # ARM NEON优化
    └── gemm_neon.cpp
```

## 🔧 已实现的Kernel

### 1. GEMM (General Matrix Multiplication)

**位置**: `cpu/gemm_cpu.cpp`

**接口**:
```cpp
namespace kernels {
class GEMMKernel {
    // C = A @ B
    template<typename T>
    static void gemm_nn(const T* A, const T* B, T* C, 
                       int M, int N, int K);
    
    // C = A @ B^T  
    template<typename T>
    static void gemm_nt(const T* A, const T* B, T* C,
                       int M, int N, int K);
};
}
```

**使用示例**:
```cpp
// Conv2D中使用
kernels::GEMMKernel::gemm_nn<float>(
    weight, col_buffer, output, 
    C_out, H_out*W_out, C_in*kH*kW
);

// Linear中使用
kernels::GEMMKernel::gemm_nt<float>(
    input, weight, output,
    batch_size, out_features, in_features
);
```

**性能特点**:
- ✅ 循环展开（4元素/次）
- ⏳ 未来：AVX2/AVX512向量化
- ⏳ 未来：OpenMP并行化
- ⏳ 未来：Cache blocking优化

### 2. Im2Col (Image to Column)

**位置**: `cpu/im2col_cpu.cpp`

**接口**:
```cpp
namespace kernels {
class Im2ColKernel {
    template<typename T>
    static void im2col(
        const T* input, T* col_buffer,
        int channels, int height, int width,
        int kernel_h, int kernel_w,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w,
        int out_height, int out_width
    );
};
}
```

**使用示例**:
```cpp
// Conv2D中使用
kernels::Im2ColKernel::im2col<float>(
    input_n, col_buffer.data(),
    C_in, H_in, W_in,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    H_out, W_out
);
```

## 🚀 性能优化路径

### CPU优化

1. **当前（v1.0）**: 朴素实现
   - 循环展开
   - 缓存友好的访问模式

2. **短期（v1.1）**: SIMD向量化
   ```cpp
   // AVX2: 8个float/次
   __m256 a = _mm256_load_ps(&A[i]);
   __m256 b = _mm256_load_ps(&B[i]);
   __m256 c = _mm256_fmadd_ps(a, b, c);
   ```

3. **中期（v1.2）**: BLAS集成
   ```cpp
   #ifdef USE_OPENBLAS
       cblas_sgemm(...);
   #else
       gemm_cpu(...);
   #endif
   ```

4. **长期（v2.0）**: 自适应优化
   ```cpp
   // 运行时选择最优kernel
   if (M > 1024 && N > 1024)
       gemm_blas(...);      // 大矩阵用BLAS
   else
       gemm_cpu_avx2(...);  // 小矩阵用AVX2
   ```

### GPU优化

```cpp
// CUDA实现（未来）
template<>
void GEMMKernel::gemm_nn<float>(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    KernelBackend::CUDA) {
    
    // Option 1: cuBLAS
    cublasSgemm(handle, ...);
    
    // Option 2: 自定义CUDA kernel
    gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
    
    // Option 3: CUTLASS模板库
    cutlass::gemm::device::Gemm<...> gemm_op;
    gemm_op(M, N, K, ...);
}
```

## 📊 Benchmark目标

| 操作 | 当前 | v1.1 (AVX2) | v1.2 (BLAS) | v2.0 (CUDA) |
|------|------|-------------|-------------|-------------|
| GEMM (1024x1024) | 100ms | 25ms | 10ms | 1ms |
| Conv2D (224x224x64) | 500ms | 125ms | 50ms | 5ms |

## 🔌 扩展指南

### 添加新的CPU优化版本

1. 创建文件: `cpu/gemm_cpu_avx2.cpp`
2. 实现优化版本:
```cpp
namespace kernels {
namespace cpu {
namespace avx2 {
    template<typename T>
    void gemm_nn_impl(...) {
        // AVX2实现
    }
}
}
}
```

3. 更新dispatcher:
```cpp
template<typename T>
void GEMMKernel::gemm_nn(..., KernelBackend backend) {
    switch (backend) {
        case KernelBackend::CPU_AVX2:
            cpu::avx2::gemm_nn_impl<T>(...);
            break;
        default:
            cpu::gemm_nn_impl<T>(...);
    }
}
```

### 添加CUDA支持

1. 创建文件: `cuda/gemm_cuda.cu`
2. CMakeLists.txt:
```cmake
if(USE_CUDA)
    enable_language(CUDA)
    target_sources(mini_infer_kernels PRIVATE
        cuda/gemm_cuda.cu
        cuda/im2col_cuda.cu
    )
    target_compile_options(mini_infer_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75>
    )
endif()
```

## 📚 参考资料

- [How to Optimize GEMM](https://github.com/flame/how-to-optimize-gemm)
- [Caffe Im2Col](https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html)
- [CUTLASS](https://github.com/NVIDIA/cutlass)

## 🤝 贡献指南

欢迎贡献新的kernel实现！请确保：

1. ✅ 保持接口一致性
2. ✅ 添加单元测试
3. ✅ 性能基准测试
4. ✅ 文档和注释
5. ✅ 跨平台兼容性
