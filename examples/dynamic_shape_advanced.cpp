#include "mini_infer/runtime/engine.h"
#include "mini_infer/runtime/optimization_profile.h"
#include "mini_infer/graph/graph.h"
#include "mini_infer/operators/conv2d.h"
#include "mini_infer/operators/relu.h"
#include "mini_infer/operators/pooling.h"
#include "mini_infer/utils/logger.h"
#include <chrono>
#include <unordered_map>

using namespace mini_infer;

void run_inference(
    runtime::Engine& engine,
    const std::string& test_name,
    const core::Shape& input_shape
) {
    MI_LOG_INFO("");
    MI_LOG_INFO("----------------------------------------");
    MI_LOG_INFO("Test: " + test_name);
    MI_LOG_INFO("Input shape: " + input_shape.to_string());
    
    // Create input tensor
    auto input = std::make_shared<core::Tensor>(input_shape, core::DataType::FLOAT32);
    
    // Fill with dummy data
    auto* data = static_cast<float*>(input->data());
    if (data) {
        auto numel = static_cast<size_t>(input->shape().numel());
        for (size_t i = 0; i < numel; ++i) {
            data[i] = 0.1f;
        }
    }
    
    // Prepare inputs
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> inputs;
    inputs["input"] = input;
    
    // Run forward
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> outputs;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto status = engine.forward(inputs, outputs);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (status == core::Status::SUCCESS) {
        MI_LOG_INFO("[SUCCESS] Inference completed in " + 
                   std::to_string(duration.count() / 1000.0) + " ms");
        
        // Show output shape
        if (outputs.find("pool1") != outputs.end()) {
            auto output_shape = outputs["pool1"]->shape();
            MI_LOG_INFO("Output shape: " + output_shape.to_string());
        }
    } else {
        MI_LOG_ERROR("[FAILED] Inference failed!");
    }
}

int main() {
    MI_LOG_INFO("========================================");
    MI_LOG_INFO("Dynamic Shape Advanced Demo");
    MI_LOG_INFO("========================================");
    
    // Step 1: Create graph
    MI_LOG_INFO("");
    MI_LOG_INFO("[Step 1] Creating CNN graph...");
    
    auto graph = std::make_shared<graph::Graph>();
    
    // Input
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
    conv_param.use_bias = false;  // This demo omits bias tensor
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
    
    // Connect
    static_cast<void>(graph->connect("input", "conv1"));
    static_cast<void>(graph->connect("conv1", "relu1"));
    static_cast<void>(graph->connect("relu1", "pool1"));
    
    graph->set_inputs({"input"});
    graph->set_outputs({"pool1"});
    
    MI_LOG_INFO("Graph created: input -> conv -> relu -> pool");
    
    // Step 2: Create optimization profile
    MI_LOG_INFO("");
    MI_LOG_INFO("[Step 2] Creating optimization profile...");
    
    auto profile = std::make_shared<runtime::OptimizationProfile>();
    profile->set_shape_range("input",
        core::Shape({1, 3, 224, 224}),   // min: single image, low res
        core::Shape({4, 3, 384, 384}),   // opt: small batch, medium res
        core::Shape({8, 3, 512, 512})    // max: large batch, high res
    );
    
    MI_LOG_INFO("Profile range:");
    MI_LOG_INFO("  Min: [1, 3, 224, 224]");
    MI_LOG_INFO("  Opt: [4, 3, 384, 384]");
    MI_LOG_INFO("  Max: [8, 3, 512, 512]");
    
    // Step 3: Build engine
    MI_LOG_INFO("");
    MI_LOG_INFO("[Step 3] Building engine with dynamic shape support...");
    
    runtime::EngineConfig config;
    config.enable_dynamic_shapes = true;
    config.optimization_profile = profile;
    config.enable_profiling = true;
    config.enable_graph_optimization = true;
    config.enable_memory_planning = true;
    
    runtime::Engine engine(config);
    auto status = engine.build(graph);
    
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("Failed to build engine");
        return 1;
    }
    
    MI_LOG_INFO("Engine built successfully!");
    
    // Step 4: Test with various input shapes
    MI_LOG_INFO("");
    MI_LOG_INFO("[Step 4] Running inference with different shapes...");
    
    // Test 1: Min shape
    run_inference(engine, "Min shape (batch=1)", core::Shape({1, 3, 224, 224}));
    
    // Test 2: Optimal shape
    run_inference(engine, "Optimal shape (batch=4)", core::Shape({4, 3, 384, 384}));
    
    // Test 3: Different batch size
    run_inference(engine, "Different batch (batch=2)", core::Shape({2, 3, 256, 256}));
    
    // Test 4: Max shape
    run_inference(engine, "Max shape (batch=8)", core::Shape({8, 3, 512, 512}));
    
    // Test 5: Back to min shape (test caching)
    run_inference(engine, "Back to min (cache test)", core::Shape({1, 3, 224, 224}));
    
    // Test 6: Out of range (should fail)
    MI_LOG_INFO("");
    MI_LOG_INFO("----------------------------------------");
    MI_LOG_INFO("Test: Out of range (should fail)");
    MI_LOG_INFO("Input shape: [16, 3, 600, 600]");
    
    auto invalid_input = std::make_shared<core::Tensor>(
        core::Shape({16, 3, 600, 600}),
        core::DataType::FLOAT32
    );
    
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> invalid_inputs;
    invalid_inputs["input"] = invalid_input;
    
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> invalid_outputs;
    status = engine.forward(invalid_inputs, invalid_outputs);
    
    if (status != core::Status::SUCCESS) {
        MI_LOG_INFO("[EXPECTED] Validation correctly rejected out-of-range shape");
    } else {
        MI_LOG_WARNING("[UNEXPECTED] Out-of-range shape was accepted!");
    }
    
    // Summary
    MI_LOG_INFO("");
    MI_LOG_INFO("========================================");
    MI_LOG_INFO("Summary");
    MI_LOG_INFO("========================================");
    
    const auto& plan = engine.get_memory_plan();
    MI_LOG_INFO("Memory Planning:");
    MI_LOG_INFO("  Original:  " + std::to_string(plan.original_memory / 1024.0) + " KB");
    MI_LOG_INFO("  Optimized: " + std::to_string(plan.total_memory / 1024.0) + " KB");
    MI_LOG_INFO("  Saving:    " + std::to_string(plan.memory_saving_ratio * 100.0f) + "%");
    
    MI_LOG_INFO("");
    MI_LOG_INFO("[SUCCESS] Dynamic Shape Advanced Demo Completed!");
    MI_LOG_INFO("========================================");
    
    return 0;
}

