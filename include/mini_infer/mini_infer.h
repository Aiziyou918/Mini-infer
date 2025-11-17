#pragma once

/**
 * @file mini_infer.h
 * @brief Mini-Infer 主头文件 - 包含所有公共 API
 */

// Core
#include "mini_infer/core/tensor.h"
#include "mini_infer/core/allocator.h"
#include "mini_infer/core/types.h"

// Backends
#include "mini_infer/backends/backend.h"
#include "mini_infer/backends/cpu_backend.h"

// Operators
#include "mini_infer/operators/operator.h"
#include "mini_infer/operators/conv2d.h"
#include "mini_infer/operators/relu.h"
#include "mini_infer/operators/linear.h"

// Graph
#include "mini_infer/graph/node.h"
#include "mini_infer/graph/graph.h"

// Runtime
#include "mini_infer/runtime/engine.h"

// Utils
#include "mini_infer/utils/logger.h"

/**
 * @mainpage Mini-Infer Documentation
 * 
 * Mini-Infer 是一个轻量级的深度学习推理框架，类似于 TensorRT。
 * 
 * @section intro_sec 简介
 * 
 * Mini-Infer 提供：
 * - 高性能的张量计算
 * - 灵活的后端支持（CPU、CUDA）
 * - 完整的计算图表示
 * - 易用的 API
 * 
 * @section install_sec 安装
 * 
 * 使用 CMake 构建：
 * @code
 * mkdir build && cd build
 * cmake ..
 * cmake --build .
 * @endcode
 * 
 * @section usage_sec 使用示例
 * 
 * @code
 * #include "mini_infer/mini_infer.h"
 * 
 * using namespace mini_infer;
 * 
 * // 创建张量
 * core::Shape shape({1, 3, 224, 224});
 * auto tensor = core::Tensor::create(shape, core::DataType::FLOAT32);
 * 
 * // 创建后端
 * auto backend = backends::BackendFactory::get_default_backend();
 * 
 * // 构建计算图
 * auto graph = std::make_shared<graph::Graph>();
 * // ...
 * 
 * // 执行推理
 * runtime::EngineConfig config;
 * runtime::Engine engine(config);
 * engine.build(graph);
 * @endcode
 */

namespace mini_infer {

/**
 * @brief 获取版本信息
 */
inline const char* get_version() {
    return "0.1.0";
}

/**
 * @brief 获取构建信息
 */
inline const char* get_build_info() {
    return "Mini-Infer v0.1.0 - Built with C++17";
}

} // namespace mini_infer

