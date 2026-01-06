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
│     Plugin System (Operators)       │  插件层 (IPlugin, PluginRegistry)
├──────────────┬──────────────────────┤
│  CPU Plugins │    CUDA Plugins      │  设备特定插件
├──────────────┴──────────────────────┤
│   Kernels (GEMM, Im2Col, Bias...)   │  底层计算原语
├─────────────────────────────────────┤
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

### 3. Plugin System (插件化算子系统)

**职责**: 实现算子逻辑，支持多设备异构执行。采用 TensorRT-style 的 Plugin 架构。

#### 核心组件

*   `IPlugin` (接口):
    *   **标准化接口**: 定义算子必须实现的方法。
    *   **形状推导**: `infer_output_shapes()` 计算输出形状。
    *   **执行**: `enqueue()` 执行实际计算。
    *   **工作空间**: `get_workspace_size()` 声明临时内存需求。
    *   **克隆**: `clone()` 支持多 Context 并发。

*   `PluginRegistry` (注册表):
    *   **Dispatch Key**: `{OpType, DeviceType}` 二元组。
    *   **单例模式**: 全局唯一的插件工厂。
    *   **宏注册**: `REGISTER_PLUGIN_SIMPLE(PluginClass, "Name", OpType, Device)`。

*   `CPUPlugin` / `CUDAPlugin` (CRTP 基类):
    *   **编译期多态**: 使用 CRTP 避免虚函数开销。
    *   **设备识别**: 自动返回正确的 DeviceType。
    *   **参数管理**: 统一的 `set_param()` / `get_param_ptr()` 接口。

#### 插件实现示例

```cpp
// ReLU CPU 插件
class ReLUCPUPlugin : public SimpleCPUPlugin<ReLUCPUPlugin> {
public:
    const char* get_plugin_type() const noexcept override { return "Relu"; }
    core::OpType get_op_type() const noexcept override { return core::OpType::kRELU; }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {
        output_shapes = input_shapes;  // ReLU 保持形状不变
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        // 执行 ReLU 计算
        const float* in = static_cast<const float*>(inputs[0]->data());
        float* out = static_cast<float*>(outputs[0]->data());
        for (size_t i = 0; i < inputs[0]->shape().numel(); ++i) {
            out[i] = std::max(0.0f, in[i]);
        }
        return core::Status::SUCCESS;
    }
};

// 一行代码注册
REGISTER_PLUGIN_SIMPLE(ReLUCPUPlugin, "Relu", kRELU, CPU)
```

### 4. Kernels 模块 (底层计算原语)

**职责**: 提供高性能的底层计算函数，被 Plugin 调用。

*   `GEMMKernel`: 通用矩阵乘法 (C = A @ B)。
*   `Im2ColKernel`: 卷积的图像到列变换。
*   `BiasKernel`: 偏置加法。
*   `TransposeKernel`: 矩阵转置。

Kernel 层是纯粹的计算函数，不包含任何算子逻辑或形状推导。

### 5. Backends 模块 (执行环境)

**职责**: 管理设备执行环境。

*   `DeviceContext`:
    *   **Execution Environment**: 管理线程池 (CPU) 或 Stream/Handle (GPU)。
    *   **TLS Injection**: 通过 Thread Local Storage 注入到 Plugin 中。

### 6. Graph 模块 (拓扑结构)

**职责**: 描述计算逻辑。

*   `Node`:
    *   **Port-Based**: 支持多输入多输出 (MIMO)，如 Split/Concat/LSTM。
    *   **OpType Caching**: 优化遍历速度。
*   `Graph`:
    *   **Topological Sort**: 保证执行顺序。
    *   **Validation**: 环检测。

### 7. Importers (前端)

*   `OnnxParser`:
    *   **Pimpl IDIOM**: 隔离 Protobuf 依赖，保证 ABI 兼容性。
    *   **ModelImporter**: 将 ONNX 节点映射为 Graph Node 连接。

## 关键技术特性

### 1. 静态内存规划 (Static Memory Planning)
采用 **Linear Scan** 算法，在 Graph 构建阶段计算所有 Tensor 的生命周期。我们将所有中间张量压缩到**一块连续内存**中。
*   **收益**: 内存碎片率接近 0，内存分配开销为 O(1)。

### 2. 零拷贝设计 (Zero-Copy)
*   **Tensor View**: Reshape/Slice 操作只修改 Metadata。
*   **Input Binding**: 支持用户传入外部指针直接作为输入 Tensor，避免 Host-to-Device 拷贝。

### 3. 插件化算子系统 (Plugin Architecture)
*   **解耦合**: 算子元数据与计算逻辑分离，Plugin 负责所有计算。
*   **易扩展**: 添加新算子只需实现 IPlugin 接口并注册。
*   **多设备**: 同一算子可有 CPU 和 CUDA 两种实现，运行时自动选择。
*   **CRTP 优化**: 使用编译期多态减少虚函数开销。

### 4. TensorRT-style 权重预加载
*   **Build Time Loading**: 权重在构建阶段加载到设备内存。
*   **Zero Runtime Overhead**: 推理时无需额外的内存拷贝。

### 5. 下一步规划 (Roadmap)

*   [ ] **Optimization**: 为 CPU Kernel 引入 AVX2/AVX-512 指令集优化。
*   [x] **GPU Support**: 实现 `CUDADeviceContext` 和 CUDA Plugins。
*   [ ] **Quantization**: 支持 INT8 量化推理。
*   [ ] **Dynamic Shape**: 完善动态形状推理支持。
