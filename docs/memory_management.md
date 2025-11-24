# 内存管理系统

## 概述

Mini-Infer 使用统一的内存分配器系统 (`CPUAllocator`) 来管理所有内存分配，确保：
- 统一的内存管理接口
- 内存使用追踪和统计
- 线程安全
- 为未来的 GPU 内存管理预留接口

## 架构

```
┌─────────────────────────────────────┐
│         Application Code            │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│    Tensor / Buffer (RAII Wrapper)   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│         CPUAllocator                │
│  (Singleton, Thread-Safe)           │
│  - Memory Tracking                  │
│  - Peak Usage Statistics            │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      std::malloc / std::free        │
└─────────────────────────────────────┘
```

## 核心组件

### 1. CPUAllocator (单例模式)

**功能**：
- 分配和释放 CPU 内存
- 追踪所有活跃的内存分配
- 统计当前和峰值内存使用

**API**：
```cpp
class CPUAllocator : public Allocator {
public:
    // 分配内存
    void* allocate(size_t size) override;
    
    // 释放内存
    void deallocate(void* ptr) override;
    
    // 当前已分配的总内存（字节）
    size_t total_allocated() const override;
    
    // 峰值内存使用（字节）
    size_t peak_allocated() const;
    
    // 活跃分配的数量
    size_t allocation_count() const;
    
    // 获取单例实例
    static CPUAllocator* get_instance();
};
```

**线程安全**：
- 使用 `std::mutex` 保护内部状态
- 所有公共方法都是线程安全的

### 2. Tensor (使用 CPUAllocator)

**自动内存管理**：
```cpp
// 创建 Tensor 时自动使用 CPUAllocator
auto tensor = Tensor::create({8, 64, 56, 56}, DataType::FLOAT32);

// 内部实现
void Tensor::allocate() {
    size_t bytes = size_in_bytes();
    void* ptr = CPUAllocator::get_instance()->allocate(bytes);
    
    // 使用 shared_ptr 的 deleter 自动释放
    data_ = std::shared_ptr<void>(ptr, [](void* p) {
        CPUAllocator::get_instance()->deallocate(p);
    });
}
```

### 3. Buffer<T> (临时缓冲区)

**RAII 封装**：
```cpp
// 使用示例
{
    // 分配 1MB 的 float buffer
    core::Buffer<float> buffer(1024 * 1024 / sizeof(float));
    
    float* data = buffer.data();
    // 使用 data...
    
} // buffer 离开作用域，自动释放内存
```

**特性**：
- RAII 自动管理生命周期
- 禁止拷贝，支持移动
- 自动使用 CPUAllocator
- 零初始化

## 使用场景

### 1. Tensor 数据存储

```cpp
// 输入 Tensor
auto input = Tensor::create({N, C, H, W}, DataType::FLOAT32);

// 权重 Tensor
auto weight = Tensor::create({C_out, C_in, kh, kw}, DataType::FLOAT32);

// 内存由 CPUAllocator 统一管理
```

### 2. 算子临时缓冲区

#### Conv2D 的 col_buffer

```cpp
// 旧方式（不推荐）
std::vector<float> col_buffer(size);  // 使用 malloc，不受管理

// 新方式（推荐）
core::Buffer<float> col_buffer(size);  // 使用 CPUAllocator
```

**优势**：
- 统一的内存追踪
- 可以监控算子的内存使用
- 自动清理，防止内存泄漏

### 3. 内存统计

```cpp
auto allocator = CPUAllocator::get_instance();

// 运行前
size_t before = allocator->total_allocated();

// 运行推理
model->forward(inputs, outputs);

// 运行后
size_t after = allocator->total_allocated();
size_t peak = allocator->peak_allocated();

std::cout << "Memory delta: " << (after - before) / 1024.0 / 1024.0 << " MB\n";
std::cout << "Peak usage: " << peak / 1024.0 / 1024.0 << " MB\n";
std::cout << "Active allocations: " << allocator->allocation_count() << "\n";
```

## 内存追踪机制

### 实现细节

```cpp
class CPUAllocator {
private:
    mutable std::mutex mutex_;
    std::unordered_map<void*, size_t> allocations_;  // ptr -> size
    size_t total_allocated_{0};
    size_t peak_allocated_{0};
};
```

### 分配流程

```
allocate(size)
    ↓
malloc(size)
    ↓
记录: allocations_[ptr] = size
    ↓
更新: total_allocated_ += size
    ↓
更新: peak_allocated_ = max(peak, total)
    ↓
返回 ptr
```

### 释放流程

```
deallocate(ptr)
    ↓
查找: allocations_[ptr] → size
    ↓
更新: total_allocated_ -= size
    ↓
删除: allocations_.erase(ptr)
    ↓
free(ptr)
```

## 性能考虑

### 开销分析

1. **空间开销**
   - 每个分配需要存储指针和大小: `sizeof(void*) + sizeof(size_t)` ≈ 16 bytes
   - unordered_map 开销
   - 对于大内存分配（MB 级别），这个开销可以忽略

2. **时间开销**
   - Mutex 锁定/解锁
   - Hash map 查找/插入/删除
   - 对于大内存分配（推理中主要场景），开销相对很小

### 优化策略

1. **减少小内存分配**
   - 批量分配大块内存
   - 重用 Buffer

2. **内存池（未来优化）**
   ```cpp
   // 未来可以添加内存池层
   class MemoryPool {
       // 预分配大块内存
       // 快速分配小块内存
   };
   ```

## 与 TensorRT 的对比

| 特性 | TensorRT | Mini-Infer |
|------|----------|------------|
| **统一管理** | ✅ ICudaEngine | ✅ CPUAllocator |
| **内存追踪** | ✅ | ✅ |
| **峰值统计** | ✅ | ✅ |
| **内存池** | ✅ | ❌ (未来) |
| **GPU 支持** | ✅ | ❌ (预留接口) |
| **多线程** | ✅ | ✅ |

## 调试和诊断

### 检测内存泄漏

```cpp
void test_memory_leak() {
    auto allocator = CPUAllocator::get_instance();
    
    size_t initial_count = allocator->allocation_count();
    size_t initial_size = allocator->total_allocated();
    
    {
        // 测试代码
        auto tensor = Tensor::create({1000, 1000}, DataType::FLOAT32);
        // ...
    } // tensor 应该被自动释放
    
    size_t final_count = allocator->allocation_count();
    size_t final_size = allocator->total_allocated();
    
    if (final_count != initial_count) {
        std::cerr << "Memory leak detected: "
                  << (final_count - initial_count) << " allocations leaked\n";
    }
    
    if (final_size != initial_size) {
        std::cerr << "Memory leak: "
                  << (final_size - initial_size) << " bytes leaked\n";
    }
}
```

### 内存使用分析

```cpp
class MemoryProfiler {
public:
    void start() {
        allocator_ = CPUAllocator::get_instance();
        start_size_ = allocator_->total_allocated();
        start_count_ = allocator_->allocation_count();
    }
    
    void report(const std::string& label) {
        size_t current = allocator_->total_allocated();
        size_t count = allocator_->allocation_count();
        size_t peak = allocator_->peak_allocated();
        
        std::cout << "[" << label << "]\n"
                  << "  Current: " << current / 1024.0 / 1024.0 << " MB\n"
                  << "  Delta: " << (current - start_size_) / 1024.0 / 1024.0 << " MB\n"
                  << "  Peak: " << peak / 1024.0 / 1024.0 << " MB\n"
                  << "  Active allocations: " << count << "\n";
    }
    
private:
    CPUAllocator* allocator_;
    size_t start_size_;
    size_t start_count_;
};
```

## 未来扩展

### 1. CUDA 内存管理

```cpp
class CUDAAllocator : public Allocator {
public:
    void* allocate(size_t size) override {
        void* ptr;
        cudaMalloc(&ptr, size);
        // Track allocation...
        return ptr;
    }
    
    void deallocate(void* ptr) override {
        // Track deallocation...
        cudaFree(ptr);
    }
};
```

### 2. 内存池

```cpp
class MemoryPool {
    // 预分配大块内存
    // 快速分配小块
    // 减少 malloc/free 调用
};
```

### 3. 智能内存重用

```cpp
// 自动重用临时 Buffer
class BufferPool {
    std::vector<Buffer<float>> free_buffers_;
    
    Buffer<float> acquire(size_t size) {
        // 从池中获取或创建新的
    }
    
    void release(Buffer<float>&& buf) {
        // 归还到池中
    }
};
```

## 最佳实践

1. **始终使用 Tensor 或 Buffer**
   - ❌ `float* data = new float[size];`
   - ✅ `core::Buffer<float> buffer(size);`

2. **避免裸指针**
   - 让 RAII 管理生命周期
   - 防止内存泄漏

3. **监控内存使用**
   - 在开发和测试中启用内存追踪
   - 检查峰值内存使用

4. **及时释放大内存**
   - 使用局部作用域限制 Buffer 生命周期
   - 不需要时立即释放

---

**相关文件**:
- `include/mini_infer/core/allocator.h`: Allocator 接口
- `include/mini_infer/core/buffer.h`: Buffer 封装
- `include/mini_infer/core/tensor.h`: Tensor 定义
- `src/core/allocator.cpp`: CPUAllocator 实现
