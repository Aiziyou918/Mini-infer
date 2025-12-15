# 动态 Shape 支持提升计划 (TensorRT 级别)

## 🎯 目标

将 Mini-Infer 的动态 Shape 支持提升到 **TensorRT 级别**，实现：
- ✅ 任意维度动态
- ✅ Optimization Profile
- ✅ 运行时重推断
- ✅ 动态内存管理
- ✅ 性能优化

---

## 📊 当前状态 vs TensorRT

| 功能 | Mini-Infer (当前) | TensorRT | 目标 |
|-----|------------------|----------|------|
| 动态 Batch | ✅ 基础支持 | ✅ 完整支持 | 提升到完整 |
| 动态 H/W/C | ⚠️ 未测试 | ✅ 完整支持 | 新增支持 |
| Optimization Profile | ❌ 无 | ✅ Min/Opt/Max | 新增 |
| 运行时重推断 | ❌ 无 | ✅ 自动 | 新增 |
| 动态内存池 | ⚠️ 有限 | ✅ 完整 | 增强 |
| Shape 缓存 | ❌ 无 | ✅ 有 | 新增 |
| 性能分析 | ⚠️ 基础 | ✅ 详细 | 增强 |

---

## 🗺️ 实施路线图

### Phase 1: 核心基础设施 (2-3 周)

#### 1.1 Optimization Profile 系统

**目标**: 支持定义输入形状的 Min/Opt/Max 范围

**TensorRT API 参考**:
```cpp
// TensorRT
IOptimizationProfile* profile = builder->createOptimizationProfile();
profile->setDimensions("input", OptProfileSelector::kMIN, Dims4{1, 3, 224, 224});
profile->setDimensions("input", OptProfileSelector::kOPT, Dims4{4, 3, 224, 224});
profile->setDimensions("input", OptProfileSelector::kMAX, Dims4{8, 3, 512, 512});
config->addOptimizationProfile(profile);
```

**Mini-Infer 实现**:

```cpp
// include/mini_infer/runtime/optimization_profile.h
namespace mini_infer {
namespace runtime {

/**
 * @brief Shape range for optimization profile
 */
struct ShapeRange {
    core::Shape min;   // Minimum shape
    core::Shape opt;   // Optimal shape (for optimization)
    core::Shape max;   // Maximum shape
    
    ShapeRange() = default;
    ShapeRange(const core::Shape& min_, const core::Shape& opt_, const core::Shape& max_)
        : min(min_), opt(opt_), max(max_) {}
    
    bool is_valid() const;
    bool contains(const core::Shape& shape) const;
};

/**
 * @brief Optimization Profile (TensorRT-style)
 * 
 * Defines the range of valid input shapes and helps the engine
 * optimize for specific shape ranges.
 */
class OptimizationProfile {
public:
    OptimizationProfile() = default;
    
    /**
     * @brief Set shape range for an input
     * 
     * @param input_name Name of the input tensor
     * @param min Minimum shape
     * @param opt Optimal shape (for kernel selection and optimization)
     * @param max Maximum shape
     * @return Status
     */
    core::Status set_shape_range(
        const std::string& input_name,
        const core::Shape& min,
        const core::Shape& opt,
        const core::Shape& max
    );
    
    /**
     * @brief Get shape range for an input
     */
    const ShapeRange* get_shape_range(const std::string& input_name) const;
    
    /**
     * @brief Check if a set of input shapes is valid for this profile
     */
    bool is_valid_for(const std::map<std::string, core::Shape>& shapes) const;
    
    /**
     * @brief Get all input names in this profile
     */
    std::vector<std::string> get_input_names() const;
    
private:
    std::map<std::string, ShapeRange> shape_ranges_;
};

} // namespace runtime
} // namespace mini_infer
```

**实现文件**:
```cpp
// src/runtime/optimization_profile.cpp
#include "mini_infer/runtime/optimization_profile.h"

namespace mini_infer {
namespace runtime {

bool ShapeRange::is_valid() const {
    // All shapes must have same ndim
    if (min.ndim() != opt.ndim() || opt.ndim() != max.ndim()) {
        return false;
    }
    
    // For each dimension: min <= opt <= max
    for (size_t i = 0; i < min.ndim(); ++i) {
        int64_t min_dim = min[i];
        int64_t opt_dim = opt[i];
        int64_t max_dim = max[i];
        
        // Skip dynamic dimensions
        if (min_dim < 0 || opt_dim < 0 || max_dim < 0) continue;
        
        if (!(min_dim <= opt_dim && opt_dim <= max_dim)) {
            return false;
        }
    }
    
    return true;
}

bool ShapeRange::contains(const core::Shape& shape) const {
    if (shape.ndim() != min.ndim()) {
        return false;
    }
    
    for (size_t i = 0; i < shape.ndim(); ++i) {
        int64_t dim = shape[i];
        int64_t min_dim = min[i];
        int64_t max_dim = max[i];
        
        // Skip dynamic dimensions in range
        if (min_dim < 0 || max_dim < 0) continue;
        
        if (dim < min_dim || dim > max_dim) {
            return false;
        }
    }
    
    return true;
}

core::Status OptimizationProfile::set_shape_range(
    const std::string& input_name,
    const core::Shape& min,
    const core::Shape& opt,
    const core::Shape& max
) {
    ShapeRange range(min, opt, max);
    
    if (!range.is_valid()) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    shape_ranges_[input_name] = range;
    return core::Status::SUCCESS;
}

const ShapeRange* OptimizationProfile::get_shape_range(const std::string& input_name) const {
    auto it = shape_ranges_.find(input_name);
    if (it == shape_ranges_.end()) {
        return nullptr;
    }
    return &it->second;
}

bool OptimizationProfile::is_valid_for(const std::map<std::string, core::Shape>& shapes) const {
    for (const auto& [name, range] : shape_ranges_) {
        auto it = shapes.find(name);
        if (it == shapes.end()) {
            return false;  // Missing input
        }
        
        if (!range.contains(it->second)) {
            return false;  // Shape out of range
        }
    }
    
    return true;
}

std::vector<std::string> OptimizationProfile::get_input_names() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : shape_ranges_) {
        names.push_back(name);
    }
    return names;
}

} // namespace runtime
} // namespace mini_infer
```

**任务清单**:
- [ ] 创建 `OptimizationProfile` 类
- [ ] 实现 `ShapeRange` 验证逻辑
- [ ] 添加到 `EngineConfig`
- [ ] 单元测试
- [ ] 文档

**预计时间**: 3-4 天

---

#### 1.2 运行时 Shape 推断引擎

**目标**: 支持在 `forward()` 时根据实际输入重新推断形状

**TensorRT 行为**:
- 检测输入形状变化
- 自动重新推断所有中间 tensor 形状
- 更新内存分配

**Mini-Infer 实现**:

```cpp
// include/mini_infer/runtime/shape_inference_engine.h
namespace mini_infer {
namespace runtime {

/**
 * @brief Shape inference context for runtime
 * 
 * Caches shape inference results for different input shapes
 */
class ShapeInferenceEngine {
public:
    ShapeInferenceEngine() = default;
    
    /**
     * @brief Infer shapes for entire graph given input shapes
     * 
     * @param graph The computation graph
     * @param input_shapes Map of input name to shape
     * @param output_shapes Output: inferred shapes for all tensors
     * @return Status
     */
    core::Status infer_shapes(
        std::shared_ptr<graph::Graph> graph,
        const std::map<std::string, core::Shape>& input_shapes,
        std::map<std::string, core::Shape>& output_shapes
    );
    
    /**
     * @brief Check if shapes have been cached for given inputs
     */
    bool has_cached_shapes(const std::map<std::string, core::Shape>& input_shapes) const;
    
    /**
     * @brief Get cached shapes
     */
    const std::map<std::string, core::Shape>* get_cached_shapes(
        const std::map<std::string, core::Shape>& input_shapes
    ) const;
    
    /**
     * @brief Clear shape cache
     */
    void clear_cache();
    
    /**
     * @brief Get cache statistics
     */
    struct CacheStats {
        size_t total_inferences = 0;
        size_t cache_hits = 0;
        size_t cache_misses = 0;
        
        double hit_rate() const {
            return total_inferences > 0 
                ? static_cast<double>(cache_hits) / total_inferences 
                : 0.0;
        }
    };
    
    CacheStats get_cache_stats() const { return stats_; }
    
private:
    // Cache key: hash of input shapes
    struct ShapeCacheKey {
        std::map<std::string, core::Shape> shapes;
        
        bool operator==(const ShapeCacheKey& other) const;
        size_t hash() const;
    };
    
    struct ShapeCacheKeyHash {
        size_t operator()(const ShapeCacheKey& key) const {
            return key.hash();
        }
    };
    
    // Cache: input shapes -> all tensor shapes
    std::unordered_map<
        ShapeCacheKey, 
        std::map<std::string, core::Shape>,
        ShapeCacheKeyHash
    > cache_;
    
    CacheStats stats_;
};

} // namespace runtime
} // namespace mini_infer
```

**实现要点**:
```cpp
core::Status ShapeInferenceEngine::infer_shapes(
    std::shared_ptr<graph::Graph> graph,
    const std::map<std::string, core::Shape>& input_shapes,
    std::map<std::string, core::Shape>& output_shapes
) {
    stats_.total_inferences++;
    
    // Check cache first
    ShapeCacheKey key{input_shapes};
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        stats_.cache_hits++;
        output_shapes = it->second;
        return core::Status::SUCCESS;
    }
    
    stats_.cache_misses++;
    
    // Perform shape inference (topological order)
    auto sorted_nodes = graph->topological_sort();
    
    // Set input shapes
    for (const auto& [name, shape] : input_shapes) {
        output_shapes[name] = shape;
    }
    
    // Infer each node
    for (auto& node : sorted_nodes) {
        if (!node || !node->get_operator()) continue;
        
        // Collect input shapes
        std::vector<core::Shape> node_input_shapes;
        for (const auto& input_node : node->inputs()) {
            if (input_node) {
                auto it = output_shapes.find(input_node->name());
                if (it != output_shapes.end()) {
                    node_input_shapes.push_back(it->second);
                }
            }
        }
        
        // Add weight shapes
        for (const auto& tensor : node->input_tensors()) {
            if (tensor) {
                node_input_shapes.push_back(tensor->shape());
            }
        }
        
        // Infer output shapes
        std::vector<core::Shape> node_output_shapes;
        auto status = node->get_operator()->infer_shape(
            node_input_shapes, 
            node_output_shapes
        );
        
        if (status != core::Status::SUCCESS) {
            return status;
        }
        
        // Store output shapes
        if (!node_output_shapes.empty()) {
            output_shapes[node->name()] = node_output_shapes[0];
        }
    }
    
    // Cache results
    cache_[key] = output_shapes;
    
    return core::Status::SUCCESS;
}
```

**任务清单**:
- [ ] 创建 `ShapeInferenceEngine` 类
- [ ] 实现形状推断逻辑
- [ ] 实现形状缓存（hash key）
- [ ] 性能测试
- [ ] 单元测试

**预计时间**: 5-6 天

---

#### 1.3 动态内存管理器

**目标**: 支持根据实际形状动态分配和重用内存

**TensorRT 行为**:
- 根据 Optimization Profile 预分配内存池
- 运行时根据实际形状调整
- 最小化重新分配

**Mini-Infer 实现**:

```cpp
// include/mini_infer/runtime/dynamic_memory_manager.h
namespace mini_infer {
namespace runtime {

/**
 * @brief Dynamic memory manager (TensorRT-style)
 * 
 * Manages memory allocation for tensors with dynamic shapes
 */
class DynamicMemoryManager {
public:
    DynamicMemoryManager() = default;
    
    /**
     * @brief Prepare memory pools based on optimization profile
     * 
     * Pre-allocate pools based on max shapes in profile
     * 
     * @param profile Optimization profile with shape ranges
     * @param plan Static memory plan (from build time)
     * @return Status
     */
    core::Status prepare(
        const OptimizationProfile& profile,
        const MemoryPlan& plan
    );
    
    /**
     * @brief Allocate memory for actual shapes
     * 
     * Reuse pre-allocated pools if possible, otherwise allocate new
     * 
     * @param tensor_shapes Actual tensor shapes
     * @param allocations Output: allocated memory for each tensor
     * @return Status
     */
    core::Status allocate_for_shapes(
        const std::map<std::string, core::Shape>& tensor_shapes,
        std::map<std::string, std::shared_ptr<void>>& allocations
    );
    
    /**
     * @brief Get memory statistics
     */
    struct MemoryStats {
        size_t pool_capacity = 0;      // Total pool capacity
        size_t pool_used = 0;          // Currently used
        size_t peak_usage = 0;         // Peak usage
        size_t reallocations = 0;      // Number of reallocations
        
        double utilization() const {
            return pool_capacity > 0 
                ? static_cast<double>(pool_used) / pool_capacity 
                : 0.0;
        }
    };
    
    MemoryStats get_stats() const { return stats_; }
    
    /**
     * @brief Reset and clear all allocations
     */
    void reset();
    
private:
    struct MemoryPool {
        std::string name;
        size_t capacity;
        std::shared_ptr<void> data;
        std::vector<std::string> tensor_names;
    };
    
    std::vector<MemoryPool> pools_;
    MemoryStats stats_;
};

} // namespace runtime
} // namespace mini_infer
```

**任务清单**:
- [ ] 创建 `DynamicMemoryManager` 类
- [ ] 实现基于 Profile 的预分配
- [ ] 实现运行时分配策略
- [ ] 内存池复用逻辑
- [ ] 性能测试和优化

**预计时间**: 4-5 天

---

### Phase 2: Engine 集成 (1-2 周)

#### 2.1 扩展 EngineConfig

```cpp
// include/mini_infer/runtime/engine.h
struct EngineConfig {
    // ... existing fields ...
    
    // Dynamic shape support
    bool enable_dynamic_shapes = false;
    
    // Optimization profiles (can have multiple)
    std::vector<std::shared_ptr<OptimizationProfile>> optimization_profiles;
    
    // Active profile index
    int active_profile_index = 0;
    
    // Shape cache settings
    bool enable_shape_cache = true;
    size_t max_shape_cache_size = 100;
    
    // Memory management
    bool enable_dynamic_memory = true;
    size_t memory_pool_growth_factor = 2;  // 2x growth when resizing
};
```

#### 2.2 扩展 Engine 类

```cpp
class Engine {
public:
    // ... existing methods ...
    
    /**
     * @brief Add optimization profile
     * 
     * Must be called before build()
     */
    core::Status add_optimization_profile(
        std::shared_ptr<OptimizationProfile> profile
    );
    
    /**
     * @brief Set active optimization profile
     * 
     * @param index Profile index (0-based)
     */
    core::Status set_active_profile(int index);
    
    /**
     * @brief Get current active profile
     */
    const OptimizationProfile* get_active_profile() const;
    
    /**
     * @brief Get shape inference statistics
     */
    ShapeInferenceEngine::CacheStats get_shape_inference_stats() const;
    
    /**
     * @brief Get dynamic memory statistics
     */
    DynamicMemoryManager::MemoryStats get_memory_stats() const;
    
private:
    /**
     * @brief Prepare for dynamic shapes (called in build())
     */
    core::Status prepare_dynamic_shapes();
    
    /**
     * @brief Handle shape change at runtime (called in forward())
     */
    core::Status handle_shape_change(
        const std::map<std::string, std::shared_ptr<core::Tensor>>& inputs
    );
    
    // New members
    std::unique_ptr<ShapeInferenceEngine> shape_engine_;
    std::unique_ptr<DynamicMemoryManager> memory_manager_;
    std::map<std::string, core::Shape> last_input_shapes_;
};
```

#### 2.3 修改 build() 流程

```cpp
core::Status Engine::build(std::shared_ptr<graph::Graph> graph) {
    MI_LOG_INFO("[Engine] Building Engine (dynamic shape support)");
    
    // Step 1: Graph optimization
    optimize_graph();
    
    // Step 2: Topological sort
    topological_sort();
    
    // Step 3: Shape inference (using optimal shapes from profile)
    if (config_.enable_dynamic_shapes && !config_.optimization_profiles.empty()) {
        const auto& profile = config_.optimization_profiles[config_.active_profile_index];
        
        // Use optimal shapes for build-time inference
        std::map<std::string, core::Shape> opt_shapes;
        for (const auto& input_name : graph_->inputs()) {
            const auto* range = profile->get_shape_range(input_name);
            if (range) {
                opt_shapes[input_name] = range->opt;
            }
        }
        
        infer_shapes_with(opt_shapes);
    } else {
        infer_shapes();  // Traditional static inference
    }
    
    // Step 4: Memory planning
    plan_memory();
    
    // Step 5: Prepare dynamic shape support
    if (config_.enable_dynamic_shapes) {
        prepare_dynamic_shapes();
    }
    
    // Step 6: Allocate tensors
    allocate_tensors();
    
    MI_LOG_INFO("[Engine] Engine built successfully");
    return core::Status::SUCCESS;
}
```

#### 2.4 修改 forward() 流程

```cpp
TensorMap Engine::forward(const TensorMap& inputs) override {
    // Step 1: Check if input shapes changed
    bool shape_changed = false;
    std::map<std::string, core::Shape> current_shapes;
    
    for (const auto& [name, tensor] : inputs) {
        current_shapes[name] = tensor->shape();
        
        auto it = last_input_shapes_.find(name);
        if (it == last_input_shapes_.end() || 
            it->second.to_string() != tensor->shape().to_string()) {
            shape_changed = true;
        }
    }
    
    // Step 2: Handle shape change
    if (shape_changed && config_.enable_dynamic_shapes) {
        auto status = handle_shape_change(inputs);
        if (status != core::Status::SUCCESS) {
            MI_LOG_ERROR("[Engine] Failed to handle shape change");
            return {};
        }
        
        last_input_shapes_ = current_shapes;
    }
    
    // Step 3: Execute inference
    return execute_inference(inputs);
}
```

**任务清单**:
- [ ] 扩展 `EngineConfig`
- [ ] 修改 `Engine::build()`
- [ ] 修改 `Engine::forward()`
- [ ] 实现 `handle_shape_change()`
- [ ] 集成测试

**预计时间**: 5-7 天

---

### Phase 3: 算子支持增强 (1 周)

#### 3.1 确保所有算子支持动态形状

检查并更新每个算子的 `infer_shape()` 实现：

```cpp
// 示例: Conv2D
core::Status Conv2D::infer_shape(
    const std::vector<core::Shape>& input_shapes,
    std::vector<core::Shape>& output_shapes
) const {
    // Validate inputs
    if (input_shapes.size() < 2) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    const auto& input_shape = input_shapes[0];  // [N, C_in, H, W]
    const auto& weight_shape = input_shapes[1]; // [C_out, C_in, K_h, K_w]
    
    // Support dynamic dimensions
    if (input_shape.ndim() != 4 || weight_shape.ndim() != 4) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Check channel consistency (skip if dynamic)
    if (input_shape[1] > 0 && weight_shape[1] > 0) {
        if (input_shape[1] != weight_shape[1]) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }
    
    // Calculate output shape
    int64_t N = input_shape[0];  // May be -1 (dynamic)
    int64_t C_out = weight_shape[0];
    
    // Calculate spatial dimensions (handle dynamic H/W)
    int64_t H_out = -1;
    int64_t W_out = -1;
    
    if (input_shape[2] > 0 && input_shape[3] > 0) {
        H_out = (input_shape[2] + 2 * param_.padding_h - param_.kernel_h) / param_.stride_h + 1;
        W_out = (input_shape[3] + 2 * param_.padding_w - param_.kernel_w) / param_.stride_w + 1;
    }
    
    output_shapes = {core::Shape({N, C_out, H_out, W_out})};
    return core::Status::SUCCESS;
}
```

**任务清单**:
- [ ] 审查所有算子的 `infer_shape()`
- [ ] 更新以支持动态维度（-1）
- [ ] 添加详细的形状验证
- [ ] 单元测试每个算子

**预计时间**: 4-5 天

---

### Phase 4: 测试与优化 (1-2 周)

#### 4.1 单元测试

```cpp
// tests/test_dynamic_shape_advanced.cpp

TEST(DynamicShapeTest, OptimizationProfile) {
    auto profile = std::make_shared<OptimizationProfile>();
    
    profile->set_shape_range(
        "input",
        Shape({1, 3, 224, 224}),   // min
        Shape({4, 3, 384, 384}),   // opt
        Shape({8, 3, 512, 512})    // max
    );
    
    // Valid shapes
    EXPECT_TRUE(profile->is_valid_for({{"input", Shape({1, 3, 224, 224})}}));
    EXPECT_TRUE(profile->is_valid_for({{"input", Shape({4, 3, 384, 384})}}));
    EXPECT_TRUE(profile->is_valid_for({{"input", Shape({8, 3, 512, 512})}}));
    
    // Invalid shapes
    EXPECT_FALSE(profile->is_valid_for({{"input", Shape({16, 3, 224, 224})}}));  // batch too large
    EXPECT_FALSE(profile->is_valid_for({{"input", Shape({4, 3, 1024, 1024})}})); // H/W too large
}

TEST(DynamicShapeTest, RuntimeShapeInference) {
    auto graph = create_test_graph();
    ShapeInferenceEngine engine;
    
    std::map<std::string, Shape> input_shapes1 = {
        {"input", Shape({1, 3, 224, 224})}
    };
    std::map<std::string, Shape> output_shapes1;
    
    auto status = engine.infer_shapes(graph, input_shapes1, output_shapes1);
    EXPECT_EQ(status, Status::SUCCESS);
    EXPECT_GT(output_shapes1.size(), 0);
    
    // Different input shape
    std::map<std::string, Shape> input_shapes2 = {
        {"input", Shape({4, 3, 384, 384})}
    };
    std::map<std::string, Shape> output_shapes2;
    
    status = engine.infer_shapes(graph, input_shapes2, output_shapes2);
    EXPECT_EQ(status, Status::SUCCESS);
    
    // Check cache
    auto stats = engine.get_cache_stats();
    EXPECT_EQ(stats.cache_misses, 2);
    EXPECT_EQ(stats.cache_hits, 0);
    
    // Reuse first shape (cache hit)
    std::map<std::string, Shape> output_shapes3;
    status = engine.infer_shapes(graph, input_shapes1, output_shapes3);
    
    stats = engine.get_cache_stats();
    EXPECT_EQ(stats.cache_hits, 1);
}

TEST(DynamicShapeTest, DynamicMemoryAllocation) {
    DynamicMemoryManager manager;
    
    // Prepare with profile
    OptimizationProfile profile;
    profile.set_shape_range("input", 
        Shape({1, 3, 224, 224}),
        Shape({4, 3, 384, 384}),
        Shape({8, 3, 512, 512})
    );
    
    MemoryPlan plan;  // From build time
    manager.prepare(profile, plan);
    
    // Allocate for actual shapes
    std::map<std::string, Shape> shapes1 = {
        {"input", Shape({1, 3, 224, 224})},
        {"conv1", Shape({1, 64, 112, 112})}
    };
    
    std::map<std::string, std::shared_ptr<void>> allocations1;
    auto status = manager.allocate_for_shapes(shapes1, allocations1);
    EXPECT_EQ(status, Status::SUCCESS);
    
    // Different shapes
    std::map<std::string, Shape> shapes2 = {
        {"input", Shape({4, 3, 384, 384})},
        {"conv1", Shape({4, 64, 192, 192})}
    };
    
    std::map<std::string, std::shared_ptr<void>> allocations2;
    status = manager.allocate_for_shapes(shapes2, allocations2);
    EXPECT_EQ(status, Status::SUCCESS);
    
    // Check memory reuse
    auto stats = manager.get_stats();
    EXPECT_GT(stats.pool_capacity, 0);
    EXPECT_LE(stats.reallocations, 1);  // Should reuse pool
}
```

#### 4.2 集成测试

```cpp
// examples/dynamic_shape_advanced_demo.cpp

int main() {
    // 1. Load ONNX model with dynamic shapes
    OnnxParser parser;
    auto graph = parser.parse_from_file("resnet50_dynamic.onnx");
    
    // 2. Create optimization profile
    auto profile = std::make_shared<OptimizationProfile>();
    profile->set_shape_range(
        "input",
        Shape({1, 3, 224, 224}),   // min: single image
        Shape({4, 3, 384, 384}),   // opt: small batch, medium res
        Shape({16, 3, 512, 512})   // max: large batch, high res
    );
    
    // 3. Configure engine
    EngineConfig config;
    config.enable_dynamic_shapes = true;
    config.enable_shape_cache = true;
    config.enable_dynamic_memory = true;
    config.enable_profiling = true;
    config.optimization_profiles.push_back(profile);
    
    // 4. Build engine (uses optimal shapes)
    Engine engine(config);
    engine.build(graph);
    
    // 5. Run inference with different shapes
    std::vector<std::tuple<int, int, int>> test_cases = {
        {1, 224, 224},
        {2, 256, 256},
        {4, 384, 384},
        {8, 512, 512},
        {1, 224, 224},  // Repeat: should hit cache
    };
    
    for (const auto& [batch, height, width] : test_cases) {
        auto input = std::make_shared<Tensor>(
            Shape({batch, 3, height, width}),
            DataType::FLOAT32
        );
        
        // Fill with random data
        fill_random(input);
        
        MI_LOG_INFO("Testing shape: [" + std::to_string(batch) + ", 3, " +
                   std::to_string(height) + ", " + std::to_string(width) + "]");
        
        auto outputs = engine.forward({{"input", input}});
        
        // Print output shapes
        for (const auto& [name, tensor] : outputs) {
            MI_LOG_INFO("  Output: " + name + " " + tensor->shape().to_string());
        }
    }
    
    // 6. Print statistics
    auto shape_stats = engine.get_shape_inference_stats();
    MI_LOG_INFO("Shape inference cache hit rate: " + 
               std::to_string(shape_stats.hit_rate() * 100) + "%");
    
    auto memory_stats = engine.get_memory_stats();
    MI_LOG_INFO("Memory utilization: " + 
               std::to_string(memory_stats.utilization() * 100) + "%");
    MI_LOG_INFO("Memory reallocations: " + 
               std::to_string(memory_stats.reallocations));
    
    return 0;
}
```

#### 4.3 性能基准测试

```cpp
// tests/benchmark_dynamic_shape.cpp

void benchmark_dynamic_vs_static() {
    // Setup
    auto graph = load_test_model();
    
    // Test 1: Static shape (baseline)
    {
        EngineConfig config;
        config.enable_dynamic_shapes = false;
        Engine engine(config);
        engine.build(graph);
        
        auto input = create_input(Shape({4, 3, 224, 224}));
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; ++i) {
            engine.forward({{"input", input}});
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Static shape: " << duration.count() << " ms\n";
    }
    
    // Test 2: Dynamic shape (same shape every time)
    {
        EngineConfig config;
        config.enable_dynamic_shapes = true;
        Engine engine(config);
        
        auto profile = std::make_shared<OptimizationProfile>();
        profile->set_shape_range("input",
            Shape({1, 3, 224, 224}),
            Shape({4, 3, 224, 224}),
            Shape({8, 3, 224, 224})
        );
        engine.add_optimization_profile(profile);
        engine.build(graph);
        
        auto input = create_input(Shape({4, 3, 224, 224}));
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; ++i) {
            engine.forward({{"input", input}});
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Dynamic shape (cached): " << duration.count() << " ms\n";
        
        auto stats = engine.get_shape_inference_stats();
        std::cout << "Cache hit rate: " << stats.hit_rate() * 100 << "%\n";
    }
    
    // Test 3: Dynamic shape (varying shapes)
    {
        EngineConfig config;
        config.enable_dynamic_shapes = true;
        Engine engine(config);
        
        auto profile = std::make_shared<OptimizationProfile>();
        profile->set_shape_range("input",
            Shape({1, 3, 224, 224}),
            Shape({4, 3, 224, 224}),
            Shape({8, 3, 224, 224})
        );
        engine.add_optimization_profile(profile);
        engine.build(graph);
        
        std::vector<int> batches = {1, 2, 4, 8};
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; ++i) {
            int batch = batches[i % batches.size()];
            auto input = create_input(Shape({batch, 3, 224, 224}));
            engine.forward({{"input", input}});
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Dynamic shape (varying): " << duration.count() << " ms\n";
        
        auto stats = engine.get_shape_inference_stats();
        std::cout << "Cache hit rate: " << stats.hit_rate() * 100 << "%\n";
    }
}
```

**任务清单**:
- [ ] 单元测试（所有新组件）
- [ ] 集成测试（端到端流程）
- [ ] 性能基准测试
- [ ] 内存泄漏检测
- [ ] 压力测试

**预计时间**: 7-10 天

---

### Phase 5: 文档与示例 (3-4 天)

#### 5.1 用户文档

- [ ] `docs/DYNAMIC_SHAPE_TENSORRT_LEVEL.md` - 完整用户指南
- [ ] `docs/OPTIMIZATION_PROFILE_GUIDE.md` - Profile 使用指南
- [ ] API 文档更新
- [ ] 迁移指南（从静态到动态）

#### 5.2 示例代码

- [ ] `examples/dynamic_shape_basic.cpp` - 基础用法
- [ ] `examples/dynamic_shape_advanced.cpp` - 高级特性
- [ ] `examples/optimization_profile_demo.cpp` - Profile 配置
- [ ] `examples/dynamic_batch_inference.cpp` - 动态 batch 推理

---

## 📅 总体时间表

| Phase | 内容 | 预计时间 | 依赖 |
|-------|------|---------|------|
| Phase 1 | 核心基础设施 | 2-3 周 | - |
| Phase 2 | Engine 集成 | 1-2 周 | Phase 1 |
| Phase 3 | 算子支持 | 1 周 | Phase 1 |
| Phase 4 | 测试与优化 | 1-2 周 | Phase 1-3 |
| Phase 5 | 文档与示例 | 3-4 天 | Phase 1-4 |
| **总计** | | **5-7 周** | |

---

## 🎯 里程碑

### Milestone 1: 基础设施完成 (Week 3)
- ✅ OptimizationProfile 实现
- ✅ ShapeInferenceEngine 实现
- ✅ DynamicMemoryManager 实现
- ✅ 单元测试通过

### Milestone 2: 集成完成 (Week 5)
- ✅ Engine 集成动态 shape 支持
- ✅ 所有算子支持动态维度
- ✅ 集成测试通过

### Milestone 3: 性能达标 (Week 6)
- ✅ 性能开销 < 5%（相比静态）
- ✅ 缓存命中率 > 80%
- ✅ 内存利用率 > 70%

### Milestone 4: 发布就绪 (Week 7)
- ✅ 所有测试通过
- ✅ 文档完善
- ✅ 示例齐全

---

## 🚀 成功标准

完成后，Mini-Infer 应该能够：

1. **支持任意维度动态**
   ```cpp
   Shape({-1, 3, -1, -1})  // ✅ 全动态
   ```

2. **Optimization Profile**
   ```cpp
   profile->set_shape_range("input",
       Shape({1, 3, 224, 224}),
       Shape({4, 3, 384, 384}),
       Shape({8, 3, 512, 512})
   );
   ```

3. **运行时自动重推断**
   ```cpp
   engine.forward(input_224);  // Auto infer
   engine.forward(input_384);  // Auto infer
   engine.forward(input_224);  // Cache hit!
   ```

4. **高效内存管理**
   - 内存池复用
   - 最小化重分配
   - < 5% 开销

5. **详细的统计信息**
   ```cpp
   auto stats = engine.get_shape_inference_stats();
   // cache_hits, cache_misses, hit_rate
   
   auto mem_stats = engine.get_memory_stats();
   // pool_capacity, pool_used, reallocations
   ```

---

## 📊 预期性能目标

| 指标 | 目标 | 说明 |
|-----|------|------|
| Shape 推断开销 | < 1ms | 缓存命中时 |
| 首次推断开销 | < 10ms | 复杂模型 |
| 缓存命中率 | > 80% | 典型应用 |
| 内存利用率 | > 70% | 避免浪费 |
| 重分配次数 | < 5% | 典型应用中 |
| API 开销 | < 5% | vs 静态 shape |

---

## 🔄 与现有代码的兼容性

**向后兼容**：
- 默认 `enable_dynamic_shapes = false`
- 现有代码无需修改
- 逐步迁移

**迁移路径**：
```cpp
// Old (static)
Engine engine(config);
engine.build(graph);

// New (dynamic, compatible)
EngineConfig config;
config.enable_dynamic_shapes = true;
config.optimization_profiles.push_back(profile);
Engine engine(config);
engine.build(graph);
```

---

## 📚 参考资料

### TensorRT 文档
- [Dynamic Shapes](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes)
- [Optimization Profiles](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimization_profiles)
- [IExecutionContext](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_execution_context.html)

### 其他框架
- ONNX Runtime: Dynamic Shape Support
- PyTorch JIT: Dynamic Shapes
- TVM: Dynamic Shape Inference

---

## ✅ 总结

这个计划将 Mini-Infer 的动态 Shape 支持提升到 **TensorRT 级别**：

**核心特性**:
- ✅ Optimization Profile（Min/Opt/Max）
- ✅ 运行时形状重推断
- ✅ 形状缓存优化
- ✅ 动态内存管理
- ✅ 任意维度动态

**实施周期**: 5-7 周

**资源需求**: 1-2 名开发者

**风险评估**: 中等
- 主要风险：性能优化
- 缓解措施：充分的性能测试和profiling

**预期收益**:
- 🚀 支持更灵活的推理场景
- 📊 更高的内存利用率
- 🎯 达到工业级动态 shape 支持水平

让我们开始实施吧！🚀


