#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>
#include "mini_infer/runtime/execution_context.h"
#include "mini_infer/runtime/inference_plan.h"
#include "mini_infer/runtime/optimization_profile.h"
#include "mini_infer/graph/graph.h"
#include "mini_infer/operators/conv2d.h"
#include "mini_infer/operators/relu.h"

using namespace mini_infer;
using namespace mini_infer::runtime;

class DynamicShapeRuntimeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create graph: input -> conv -> relu
        graph = std::make_shared<graph::Graph>();
        
        // Input
        auto input_node = graph->create_node("input");
        auto input_tensor = std::make_shared<core::Tensor>(
            core::Shape({1, 3, 224, 224}),
            core::DataType::FLOAT32
        );
        input_node->set_output_tensors({input_tensor});
        
        // Conv
        auto conv_node = graph->create_node("conv1");
        operators::Conv2DParam conv_param;
        conv_param.stride_h = 1;
        conv_param.stride_w = 1;
        conv_param.padding_h = 0;
        conv_param.padding_w = 0;
        conv_param.use_bias = false;  // No bias tensor provided
        auto conv_op = std::make_shared<operators::Conv2D>(conv_param);
        conv_node->set_operator(conv_op);
        
        auto weight = std::make_shared<core::Tensor>(
            core::Shape({32, 3, 3, 3}),
            core::DataType::FLOAT32
        );
        conv_node->set_input_tensors({nullptr, weight});
        
        // ReLU
        auto relu_node = graph->create_node("relu1");
        auto relu_op = std::make_shared<operators::ReLU>();
        relu_node->set_operator(relu_op);
        
        // Connect
        static_cast<void>(graph->connect("input", "conv1"));
        static_cast<void>(graph->connect("conv1", "relu1"));
        
        graph->set_inputs({"input"});
        graph->set_outputs({"relu1"});
        
        // Create profile
        profile = std::make_shared<OptimizationProfile>();
        profile->set_shape_range("input",
            core::Shape({1, 3, 224, 224}),   // min
            core::Shape({4, 3, 256, 256}),   // opt
            core::Shape({8, 3, 512, 512})    // max
        );
    }
    
    std::shared_ptr<graph::Graph> graph;
    std::shared_ptr<OptimizationProfile> profile;
};

TEST_F(DynamicShapeRuntimeTest, BuildWithOptimalShape) {
    EngineConfig config;
    config.enable_dynamic_shapes = true;
    config.optimization_profile = profile;
    config.enable_profiling = true;
    
    auto plan = std::make_shared<InferencePlan>(config);
    auto status = plan->build(graph);
    
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    // Check that optimal shape was used
    auto input_node = graph->get_node("input");
    ASSERT_NE(input_node, nullptr);
    ASSERT_FALSE(input_node->output_tensors().empty());
    
    auto shape = input_node->output_tensors()[0]->shape();
    EXPECT_EQ(shape.to_string(), "[4, 3, 256, 256]");  // optimal
}

TEST_F(DynamicShapeRuntimeTest, ForwardWithDifferentBatchSizes) {
    EngineConfig config;
    config.enable_dynamic_shapes = true;
    config.optimization_profile = profile;
    
    auto plan = std::make_shared<InferencePlan>(config);
    plan->build(graph);
    auto ctx = plan->create_execution_context();
    
    // First forward with batch=1
    auto input1 = std::make_shared<core::Tensor>(
        core::Shape({1, 3, 224, 224}),
        core::DataType::FLOAT32
    );
    
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> inputs1;
    inputs1["input"] = input1;
    
    auto status = ctx->set_inputs(inputs1);
    ASSERT_EQ(status, core::Status::SUCCESS);
    status = plan->execute(ctx.get());
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    // Second forward with batch=2 (shape change)
    auto input2 = std::make_shared<core::Tensor>(
        core::Shape({2, 3, 224, 224}),
        core::DataType::FLOAT32
    );
    
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> inputs2;
    inputs2["input"] = input2;
    
    status = ctx->set_inputs(inputs2);
    ASSERT_EQ(status, core::Status::SUCCESS);
    status = plan->execute(ctx.get());
    EXPECT_EQ(status, core::Status::SUCCESS);
}

TEST_F(DynamicShapeRuntimeTest, ShapeOutOfRange) {
    EngineConfig config;
    config.enable_dynamic_shapes = true;
    config.optimization_profile = profile;
    
    auto plan = std::make_shared<InferencePlan>(config);
    plan->build(graph);
    auto ctx = plan->create_execution_context();
    
    // Try input with shape outside profile range
    auto input_too_large = std::make_shared<core::Tensor>(
        core::Shape({16, 3, 600, 600}),  // Exceeds max
        core::DataType::FLOAT32
    );
    
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> inputs;
    inputs["input"] = input_too_large;
    
    auto status = ctx->set_inputs(inputs);
    ASSERT_EQ(status, core::Status::SUCCESS);
    status = plan->execute(ctx.get());
    
    // Should fail validation
    EXPECT_NE(status, core::Status::SUCCESS);
}

TEST_F(DynamicShapeRuntimeTest, MultipleShapeChanges) {
    EngineConfig config;
    config.enable_dynamic_shapes = true;
    config.optimization_profile = profile;
    config.enable_profiling = true;
    
    auto plan = std::make_shared<InferencePlan>(config);
    plan->build(graph);
    auto ctx = plan->create_execution_context();
    
    // Test multiple different shapes
    std::vector<core::Shape> test_shapes = {
        core::Shape({1, 3, 224, 224}),
        core::Shape({2, 3, 256, 256}),
        core::Shape({4, 3, 320, 320}),
        core::Shape({1, 3, 224, 224}),  // Back to first shape
        core::Shape({8, 3, 384, 384})
    };
    
    for (const auto& shape : test_shapes) {
        auto input = std::make_shared<core::Tensor>(shape, core::DataType::FLOAT32);
        
        std::unordered_map<std::string, std::shared_ptr<core::Tensor>> inputs;
        inputs["input"] = input;
        
        auto status = ctx->set_inputs(inputs);
        ASSERT_EQ(status, core::Status::SUCCESS);
        status = plan->execute(ctx.get());
        
        EXPECT_EQ(status, core::Status::SUCCESS) 
            << "Failed with shape: " << shape.to_string();
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
