# Mini-Infer 架构设计

## 总体架构

Mini-Infer 采用 **TensorRT-like** 的分层设计，强调高性能推理和异构设备支持。

```
┌─────────────────────────────────────┐
│         Application Layer           │  应用层 (InferencePlan, ExecutionContext)
├─────────────────────────────────────┤
│          Runtime Engine             │  运行时层 (MemoryPlanner, ShapeInference)
├─────────────────────────────────────┤
│      Graph & Optimization           │  图层 (Port-based Connection)
├─────────────────────────────────────┤
│          Operators                  │  算子层 (Metadata Holder)
├─────────────────────────────────────┤
│    Registry & Dispatcher            │  分发层 (Key: OpType+Device+DataType)
├──────────────┬──────────────────────┤
│  CPU Kernels │    CUDA Kernels      │  计算核心 (IMPL)
├──────────────┴──────────────────────┤
│   Backend Context (Stream/Handle)   │  执行环境 (DeviceContext)
├─────────────────────────────────────┤
│         Core (Tensor, Storage)      │  核心层 (Zero-Copy)
└─────────────────────────────────────┘
```

## 核心模块详解

### 1. Core 模块 (基础底座)

**职责**: 提供高性能、低开销的基础数据结构。

*   `Tensor`:
    *   **View 机制**: 与 PyTorch 类似，支持 Slice/Reshape 而不拷贝数据。
    *   **Metadata**: 包含 Shape, Strides, Offset, Data Type, Device Type。
*   `Storage`:
    *   **Raw Memory**: 管理物理内存块 (void*)。
    *   **Ownership**: 通过 `shared_ptr` 管理生命周期。
*   `Allocator`:
    *   **Lock-Free**: Release 模式下无锁分配，性能极高。
    *   **Alignment**: 强制 AVX/CUDA 对齐。

### 2. Runtime 模块 (执行大脑)

**职责**: 管理推理的生命周期，实现“一次构建，多次执行”。

*   `Engine`: API 门面，隐藏内部复杂性。
*   `InferencePlan` (Immutable):
    *   持有优化后的 Graph。
    *   持有静态内存规划结果 (Memory Plan)。
    *   **线程安全**: 多线程可共享同一个 Plan。
*   `ExecutionContext` (Mutable):
    *   **Per-Request**: 每个推理请求创建一个 Context。
    *   **Memory Pool**: 根据 Plan 申请一块大内存 (`Storage`)，所有中间 Tensor 都是这块内存的 Offset View。
    *   **Device Management**: 维护 `DeviceContext` (如 CUDA Stream)。

### 3. Backends & Kernels 模块 (计算后端)

**职责**: 执行具体的数学运算，支持硬件异构。

*   `DeviceContext`:
    *   **Execution Environment**: 管理线程池 (CPU) 或 Stream/Handle (GPU)。
    *   **TLS Injection**: 通过 Thread Local Storage 注入到 Kernel 中，无需修改 Kernel 签名。
*   `KernelRegistry`:
    *   **Dispatch Key**: `{OpType, DeviceType, DataType}`。
    *   **Extensibility**: 支持静态注册新 Kernel。
*   `Kernels` (CPU/CUDA):
    *   纯粹的计算函数 (Stateless)。
    *   CPU: Im2Col + GEMM, SIMD optimized.
    *   CUDA: cuDNN / cuBLAS wrappers.

### 4. Graph 模块 (拓扑结构)

**职责**: 描述计算逻辑。

*   `Node`:
    *   **Port-Based**: 支持多输入多输出 (MIMO)，如 Split/Contact/LSTM。
    *   **OpType Caching**: 优化遍历速度。
*   `Graph`:
    *   **Topological Sort**: 保证执行顺序。
    *   **Validation**: 环检测。

### 5. Operators & Importers (前端)

*   `Operators`: 算子元数据容器 (OpParam) 和 Shape 推导逻辑。
*   `Importers (OnnxParser)`:
    *   **Pimpl IDIOM**: 隔离 Protobuf 依赖，保证 ABI 兼容性。
    *   **ModelImporter**: 将 ONNX 节点映射为 Graph Node 连接。

## 关键技术特性

### 1. 静态内存规划 (Static Memory Planning)
采用 **Linear Scan** 算法，在 Graph 构建阶段计算所有 Tensor 的生命周期。我们将所有中间张量压缩到**一块连续内存**中。
*   **收益**: 内存碎片率接近 0，内存分配开销为 O(1)。

### 2. 零拷贝设计 (Zero-Copy)
*   **Tensor View**: Reshape/Slice 操作只修改 Metadata。
*   **Input Binding**: 支持用户传入外部指针直接作为输入 Tensor，避免 Host-to-Device 拷贝。

### 3. 下一步规划 (Roadmap)

*   [ ] **Optimization**: 为 CPU Kernel 引入 AVX2/AVX-512 指令集优化。
*   [ ] **GPU Support**: 实现 `CUDADeviceContext` 和 CUDA Kernels。
*   [ ] **Quantization**: 支持 INT8 量化推理。
