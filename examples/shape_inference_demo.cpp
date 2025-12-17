/**
 * @file shape_inference_demo.cpp
 * @brief Demonstrates TensorRT-style shape inference in Mini-Infer
 * 
 * This example shows how the Engine automatically infers tensor shapes
 * during the build phase, similar to TensorRT.
 */

#include <iostream>
#include <memory>

#include "mini_infer/core/tensor.h"
#include "mini_infer/graph/graph.h"
#include "mini_infer/graph/node.h"
#include "mini_infer/operators/conv2d.h"
#include "mini_infer/operators/relu.h"
#include "mini_infer/operators/pooling.h"
#include "mini_infer/operators/flatten.h"
#include "mini_infer/operators/linear.h"
#include "mini_infer/runtime/engine.h"
#include "mini_infer/utils/logger.h"

using namespace mini_infer;

/**
 * @brief Build a simple CNN graph manually
 * Similar to: Input -> Conv -> ReLU -> Pool -> Flatten -> Linear
 */
std::shared_ptr<graph::Graph> build_simple_cnn() {
    auto graph = std::make_shared<graph::Graph>();
    
    // Input node (28x28 grayscale image)
    auto input_node = graph->create_node("input");
    auto input_tensor = std::make_shared<core::Tensor>(
        core::Shape({1, 1, 28, 28}),  // [N, C, H, W]
        core::DataType::FLOAT32
    );
    input_node->set_output_tensors({input_tensor});
    
    // Conv2D node
    auto conv_node = graph->create_node("conv1");
    operators::Conv2DParam conv_param;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.padding_h = 2;
    conv_param.padding_w = 2;
    conv_param.dilation_h = 1;
    conv_param.dilation_w = 1;
    conv_param.groups = 1;
    conv_param.use_bias = true;
    auto conv_op = std::make_shared<operators::Conv2D>(conv_param);
    conv_node->set_operator(conv_op);
    
    // Conv weights and bias
    auto conv_weight = std::make_shared<core::Tensor>(
        core::Shape({32, 1, 5, 5}), core::DataType::FLOAT32);  // [out_ch, in_ch, kh, kw]
    auto conv_bias = std::make_shared<core::Tensor>(
        core::Shape({32}), core::DataType::FLOAT32);
    conv_node->set_input_tensors({nullptr, conv_weight, conv_bias});  // data from graph, weight/bias stored
    
    // ReLU node
    auto relu_node = graph->create_node("relu1");
    auto relu_op = std::make_shared<operators::ReLU>();
    relu_node->set_operator(relu_op);
    
    // MaxPool node
    auto pool_node = graph->create_node("pool1");
    operators::PoolingParam pool_param;
    pool_param.type = operators::PoolingType::MAX;
    pool_param.kernel_h = 2;
    pool_param.kernel_w = 2;
    pool_param.stride_h = 2;
    pool_param.stride_w = 2;
    pool_param.padding_h = 0;
    pool_param.padding_w = 0;
    auto pool_op = std::make_shared<operators::Pooling>(pool_param);
    pool_node->set_operator(pool_op);
    
    // Flatten node
    auto flatten_node = graph->create_node("flatten");
    operators::FlattenParam flatten_param;
    flatten_param.axis = 1;
    auto flatten_op = std::make_shared<operators::Flatten>(flatten_param);
    flatten_node->set_operator(flatten_op);
    
    // Linear node
    auto linear_node = graph->create_node("fc");
    operators::LinearParam linear_param;
    linear_param.in_features = 32 * 14 * 14;  // After pool: 32 channels, 14x14 spatial
    linear_param.out_features = 10;
    linear_param.use_bias = true;
    auto linear_op = std::make_shared<operators::Linear>(linear_param);
    linear_node->set_operator(linear_op);
    
    // Linear weights and bias
    auto linear_weight = std::make_shared<core::Tensor>(
        core::Shape({10, 32 * 14 * 14}), core::DataType::FLOAT32);
    auto linear_bias = std::make_shared<core::Tensor>(
        core::Shape({10}), core::DataType::FLOAT32);
    linear_node->set_input_tensors({nullptr, linear_weight, linear_bias});
    
    // Connect nodes
    (void)graph->connect("input", "conv1");
    (void)graph->connect("conv1", "relu1");
    (void)graph->connect("relu1", "pool1");
    (void)graph->connect("pool1", "flatten");
    (void)graph->connect("flatten", "fc");
    
    // Set graph inputs and outputs
    graph->set_inputs({"input"});
    graph->set_outputs({"fc"});
    
    return graph;
}

/**
 * @brief Print shape information for all nodes
 */
void print_graph_shapes(graph::Graph* graph) {
    MI_LOG_INFO("========================================");
    MI_LOG_INFO("Graph Shape Information");
    MI_LOG_INFO("========================================");
    
    for (const auto& [name, node] : graph->nodes()) {
        if (!node) continue;
        
        std::string info = "Node: " + name;
        
        // Input shapes (from connected nodes)
        if (!node->inputs().empty()) {
            info += "\n  Inputs:";
            for (size_t i = 0; i < node->inputs().size(); ++i) {
                const auto& input_node = node->inputs()[i];
                if (input_node && !input_node->output_tensors().empty()) {
                    info += "\n    [" + std::to_string(i) + "] " + 
                           input_node->name() + ": " +
                           input_node->output_tensors()[0]->shape().to_string();
                }
            }
        }
        
        // Output shapes
        if (!node->output_tensors().empty()) {
            info += "\n  Outputs:";
            for (size_t i = 0; i < node->output_tensors().size(); ++i) {
                if (node->output_tensors()[i]) {
                    info += "\n    [" + std::to_string(i) + "] " +
                           node->output_tensors()[i]->shape().to_string();
                }
            }
        }
        
        MI_LOG_INFO(info);
    }
    
    MI_LOG_INFO("========================================");
}

int main() {
    MI_LOG_INFO("========================================");
    MI_LOG_INFO("Shape Inference Demo (TensorRT-style)");
    MI_LOG_INFO("========================================");
    MI_LOG_INFO("");
    
    // Step 1: Build a simple CNN graph
    MI_LOG_INFO("[Step 1] Building graph...");
    auto graph = build_simple_cnn();
    MI_LOG_INFO("Graph built with " + std::to_string(graph->nodes().size()) + " nodes");
    MI_LOG_INFO("");
    
    // Step 2: Before Engine build - shapes are only set for input and weights
    MI_LOG_INFO("[Step 2] Before shape inference:");
    print_graph_shapes(graph.get());
    MI_LOG_INFO("");
    
    // Step 3: Build Engine (automatically performs shape inference)
    MI_LOG_INFO("[Step 3] Building Engine (with shape inference)...");
    runtime::EngineConfig config;
    config.device_type = core::DeviceType::CPU;
    config.enable_graph_optimization = true;
    config.enable_memory_planning = true;
    config.enable_profiling = true;  // Enable detailed logs
    
    runtime::Engine engine(config);
    auto status = engine.build(graph);
    
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("Failed to build engine");
        return 1;
    }
    MI_LOG_INFO("");
    
    // Step 4: After Engine build - all shapes are inferred
    MI_LOG_INFO("[Step 4] After shape inference:");
    print_graph_shapes(graph.get());
    MI_LOG_INFO("");
    
    // Step 5: Show memory planning results
    const auto& memory_plan = engine.get_memory_plan();
    MI_LOG_INFO("[Step 5] Memory Planning Results:");
    MI_LOG_INFO("  Original memory:  " + 
                std::to_string(memory_plan.original_memory / 1024.0) + " KB");
    MI_LOG_INFO("  Optimized memory: " + 
                std::to_string(memory_plan.total_memory / 1024.0) + " KB");
    MI_LOG_INFO("  Memory saving:    " + 
                std::to_string(memory_plan.memory_saving_ratio * 100.0f) + "%");
    MI_LOG_INFO("  Number of pools:  " + 
                std::to_string(memory_plan.pools.size()));
    MI_LOG_INFO("");
    
    // Step 6: Verify expected shapes
    MI_LOG_INFO("[Step 6] Verifying expected shapes:");
    
    struct ExpectedShape {
        std::string node_name;
        std::vector<int64_t> expected_dims;
    };
    
    std::vector<ExpectedShape> expected_shapes = {
        {"input", {1, 1, 28, 28}},
        {"conv1", {1, 32, 28, 28}},   // Same size due to padding=2
        {"relu1", {1, 32, 28, 28}},   // ReLU preserves shape
        {"pool1", {1, 32, 14, 14}},   // Halved by stride=2
        {"flatten", {1, 6272}},       // 32 * 14 * 14 = 6272
        {"fc", {1, 10}}               // Final classification
    };
    
    bool all_correct = true;
    for (const auto& expected : expected_shapes) {
        auto node = graph->get_node(expected.node_name);
        if (!node || node->output_tensors().empty() || !node->output_tensors()[0]) {
            MI_LOG_ERROR("  [FAILED] " + expected.node_name + ": tensor not found");
            all_correct = false;
            continue;
        }
        
        const auto& actual_shape = node->output_tensors()[0]->shape();
        bool shape_match = true;
        
        if (actual_shape.ndim() != expected.expected_dims.size()) {
            shape_match = false;
        } else {
            for (size_t i = 0; i < expected.expected_dims.size(); ++i) {
                if (actual_shape[i] != expected.expected_dims[i]) {
                    shape_match = false;
                    break;
                }
            }
        }
        
        if (shape_match) {
            MI_LOG_INFO("  [SUCESS] " + expected.node_name + ": " + actual_shape.to_string());
        } else {
            core::Shape expected_shape_obj(expected.expected_dims);
            MI_LOG_ERROR("  [FAILED] " + expected.node_name + ": expected " + 
                        expected_shape_obj.to_string() + ", got " + actual_shape.to_string());
            all_correct = false;
        }
    }
    
    MI_LOG_INFO("");
    MI_LOG_INFO("========================================");
    
    if (all_correct) {
        MI_LOG_INFO("[SUCESS] All shapes inferred correctly!");
        MI_LOG_INFO("Shape inference system working as expected.");
        return 0;
    } else {
        MI_LOG_ERROR("[FAILED] Some shapes were incorrect!");
        return 1;
    }
}

