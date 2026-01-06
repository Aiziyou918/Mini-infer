# Mini-Infer Kernel Layer

## 概述

Kernel 层是 Mini-Infer 的**底层计算原语**，提供高性能的基础数学运算。在新的插件化架构中，Kernel 层被 Plugin 调用，不再直接与算子绑定。

## 架构设计

```
┌─────────────────────────────────────┐
│        Plugin Layer                 │
│   (Conv2DPlugin, LinearPlugin...)   │
└───────────────┬─────────────────────┘
                │ 调用
┌───────────────▼─────────────────────┐
│        Kernel Interface             │
│   (GEMMKernel, Im2ColKernel, ...)   │
└───────────────┬─────────────────────┘
                │
        ┌───────┴────────┐
        │                │
┌───────▼──────┐  ┌──────▼────────┐
│ CPU Kernels  │  │  CUDA Kernels │
│              │  │               │
│ - gemm_cpu   │  │ - gemm_cuda   │
│ - im2col_cpu │  │ - im2col_cuda │
│ - bias_cpu   │  │ - bias_cuda   │
└──────────────┘  └───────────────┘
```

## 目录结构

```
kernels/
├── README.md           # 本文件
├── CMakeLists.txt      # 构建配置
├── cpu/                # CPU 实现
│   ├── gemm_cpu.cpp    # GEMM 矩阵乘法
│   ├── im2col_cpu.cpp  # Im2Col 变换
│   └── bias_cpu.cpp    # 偏置加法
└── cuda/               # CUDA 实现
    ├── gemm_cuda.cu    # CUDA GEMM
    ├── im2col_cuda.cu  # CUDA Im2Col
    ├── bias_cuda.cu    # CUDA 偏置加法
    └── transpose_cuda.cu # CUDA 转置
```

## 与 Plugin 的关系

在新架构中，**Kernel 和 Plugin 的职责明确分离**：

| 层级 | 职责 | 示例 |
|------|------|------|
| **Plugin** | 算子逻辑、形状推导、参数管理 | Conv2DCPUPlugin, ReLUCUDAPlugin |
| **Kernel** | 纯粹的数学计算 | GEMMKernel, Im2ColKernel |

Plugin 负责：
- 实现 `IPlugin` 接口
- 形状推导 (`infer_output_shapes`)
- 调用 Kernel 执行计算
- 管理算子参数

Kernel 负责：
- 高性能的数学运算
- 设备特定的优化（SIMD、CUDA）
- 无状态的纯函数

## 已实现的 Kernel

### 1. GEMM (General Matrix Multiplication)

**位置**: `cpu/gemm_cpu.cpp`, `cumm_cuda.cu`

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

**使用示例** (在 Plugin 中):
```cpp
// Conv2DCPUPlugin::enqueue() 中
kernels::GEMMKernel::gemm_nn<float>(
    weight, col_buffer, output,
    C_out, H_out*W_out, C_in*kH*kW
);

// LinearCPUPlugin::enqueue() 中
kernels::GEMMKernel::gemm_nt<float>(
    inweight, output,
    batch_size, out_features, in_features
);
```

### 2. Im2Col (Image to Column)

**位置**: `cpu/im2col_cpu.cpp`, `cuda/im2col_cuda.cu`

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

### 3. Bias (偏置加法)

**位置**: `cpu/bias_cpu.cpp`, `cuda/bias_cuda.cu`

**接口**:
```cpp
namespace kernels {
class BiasKernel {
    template<typename T>
    static void add_channel_bias(
        T* data, const T* bias,
        int batch, int channels, int spatial_size
    );
};
}
```

### 4. Transpose (矩阵转置)

**位置**: `cuda/transpose_cuda.cu`

**接口**:
```cpp
namespace kernels {
class TransposeKernel {
    template<typename T>
    static void transpose_2d(
        const T* input, T* output,
        int rows, int cols
    );
};
}
```

## 性能优化

### CPU 优化

1. **当前**: 循环展开、缓存友好的访问模式
2. **计划**: AVX2/AVX-512 向量化、OpenMP 并行化

### CUDA 优化

1. **向量化访问**: 使用 `float4` 提高内存带宽利用率
2. **共享内存**: 减少全局内存访问
3. **Warp 级优化**: 利用 warp shuffle 指令

## 扩展指南

### 添加新的 CPU Kernel

1. 创建文件: `cpu/new_kernel_cpu.cpp`
2. 实现 Kernel 类:
```cpp
namespace mini_infer {
namespace kernels {

class NewKernel {
public:
    template<typename T>
    static void compute(const T* input, T* output, int size) {
        // CPU 实现
    }
};

} // namespace kernels
} // namespace mini_infer
```

3. 在 CMakeLists.txt 中添加源文件

### 添加新的 CUDA Kernel

1. 创建文件: `cuda/new_kernel_cuda.cu`
2. 实现 CUDA Kernel:
```cpp
namespace mini_infer {
namespace kernels {
namespace cuda {

template<typename T>
__global__ void new_kernel_cuda(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // CUDA 实现
    }
}

} // namespace cuda
} // namespace kernels
} // namespace mini_infer
```

## 参考资料

- [How to Optimize GEMM](https://github.com/flame/how-to-optimize-gemm)
- [Caffe Im2Col](https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
