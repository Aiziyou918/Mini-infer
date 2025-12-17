#include "mini_infer/runtime/engine.h"
#include "mini_infer/runtime/optimization_profile.h"
#include "mini_infer/graph/graph.h"
#include "mini_infer/operators/conv2d.h"
#include "mini_infer/operators/relu.h"
#include "mini_infer/operators/pooling.h"
#include "mini_infer/utils/logger.h"

using namespace mini_infer;

int main() {
    MI_LOG_INFO("========================================");
    MI_LOG_INFO("Dynamic Shape Basic Demo");
    MI_LOG_INFO("========================================");
    
    // Step 1: Create a simple CNN graph
    auto graph = std::make_shared<graph::Graph>();
    
    // Input node
    auto input_node = graph->create_node("input");
    auto input_tensor = std::make_shared<core::Tensor>(
        core::Shape({1, 3, 224, 224}),
        core::DataType::FLOAT32
    );
    input_node->set_output_tensors({input_tensor});
    
    // Conv2D
    auto conv_node = graph->create_node("conv1");
    operators::Conv2DParam conv_param;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.padding_h = 1;
    conv_param.padding_w = 1;
    conv_param.use_bias = false;
    auto conv_op = std::make_shared<operators::Conv2D>(conv_param);
    conv_node->set_operator(conv_op);
    
    auto conv_weight = std::make_shared<core::Tensor>(
        core::Shape({64, 3, 3, 3}),
        core::DataType::FLOAT32
    );
    conv_node->set_input_tensors({nullptr, conv_weight});
    
    // ReLU
    auto relu_node = graph->create_node("relu1");
    auto relu_op = std::make_shared<operators::ReLU>();
    relu_node->set_operator(relu_op);
    
    // MaxPool
    auto pool_node = graph->create_node("pool1");
    operators::PoolingParam pool_param;
    pool_param.type = operators::PoolingType::MAX;
    pool_param.kernel_h = 2;
    pool_param.kernel_w = 2;
    pool_param.stride_h = 2;
    pool_param.stride_w = 2;
    auto pool_op = std::make_shared<operators::Pooling>(pool_param);
    pool_node->set_operator(pool_op);
    
    // Connect nodes
    graph->connect("input", "conv1");
    graph->connect("conv1", "relu1");
    graph->connect("relu1", "pool1");
    
    graph->set_inputs({"input"});
    graph->set_outputs({"pool1"});
    
    // Step 2: Create optimization profile
    MI_LOG_INFO("");
    MI_LOG_INFO("[Step 1] Creating optimization profile...");
    
    auto profile = std::make_shared<runtime::OptimizationProfile>();
    
    auto status = profile->set_shape_range("input",
        core::Shape({1, 3, 224, 224}),   // min: single image, low res
        core::Shape({4, 3, 384, 384}),   // opt: small batch, medium res
        core::Shape({8, 3, 512, 512})    // max: large batch, high res
    );
    
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("Failed to set shape range");
        return 1;
    }
    
    MI_LOG_INFO("Profile created:");
    MI_LOG_INFO("  Min: [1, 3, 224, 224]");
    MI_LOG_INFO("  Opt: [4, 3, 384, 384]");
    MI_LOG_INFO("  Max: [8, 3, 512, 512]");
    
    // Step 3: Build engine with dynamic shape support
    MI_LOG_INFO("");
    MI_LOG_INFO("[Step 2] Building engine with dynamic shape support...");
    
    runtime::EngineConfig config;
    config.enable_dynamic_shapes = true;
    config.optimization_profile = profile;
    config.enable_profiling = true;
    config.enable_graph_optimization = true;
    config.enable_memory_planning = true;
    
    runtime::Engine engine(config);
    status = engine.build(graph);
    
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("Failed to build engine");
        return 1;
    }
    
    // Step 4: Check results
    MI_LOG_INFO("");
    MI_LOG_INFO("[Step 3] Checking results...");
    
    auto input_check = graph->get_node("input");
    if (input_check && !input_check->output_tensors().empty()) {
        auto shape = input_check->output_tensors()[0]->shape();
        MI_LOG_INFO("Input shape after build: " + shape.to_string());
        MI_LOG_INFO("Expected optimal shape: [4, 3, 384, 384]");
        
        if (shape.to_string() == "[4, 3, 384, 384]") {
            MI_LOG_INFO("[SUCCESS] Engine used optimal shape from profile!");
        } else {
            MI_LOG_WARNING("[WARNING] Shape doesn't match optimal");
        }
    }
    
    // Step 5: Show memory planning results
    MI_LOG_INFO("");
    MI_LOG_INFO("[Step 4] Memory planning results:");
    
    const auto& plan = engine.get_memory_plan();
    MI_LOG_INFO("  Original memory:  " + 
               std::to_string(plan.original_memory / 1024.0) + " KB");
    MI_LOG_INFO("  Optimized memory: " + 
               std::to_string(plan.total_memory / 1024.0) + " KB");
    MI_LOG_INFO("  Memory saving:    " + 
               std::to_string(plan.memory_saving_ratio * 100.0f) + "%");
    MI_LOG_INFO("  Number of pools:  " + 
               std::to_string(plan.pools.size()));
    
    MI_LOG_INFO("");
    MI_LOG_INFO("========================================");
    MI_LOG_INFO("[SUCCESS] Dynamic Shape Basic Demo Completed!");
    MI_LOG_INFO("========================================");
    
    return 0;
}


