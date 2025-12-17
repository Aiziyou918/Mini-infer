#include <gtest/gtest.h>
#include "mini_infer/runtime/engine.h"
#include "mini_infer/runtime/optimization_profile.h"
#include "mini_infer/graph/graph.h"
#include "mini_infer/operators/conv2d.h"
#include "mini_infer/operators/relu.h"

using namespace mini_infer;
using namespace mini_infer::runtime;

TEST(EngineWithProfileTest, BasicUsage) {
    // Create a simple graph: input -> conv -> relu
    auto graph = std::make_shared<graph::Graph>();
    
    // Input node
    auto input_node = graph->create_node("input");
    auto input_tensor = std::make_shared<core::Tensor>(
        core::Shape({1, 3, 224, 224}),  // Initial shape
        core::DataType::FLOAT32
    );
    input_node->set_output_tensors({input_tensor});
    
    // Conv node
    auto conv_node = graph->create_node("conv1");
    operators::Conv2DParam conv_param;
    conv_param.stride_h = 1;
    conv_param.stride_w = 1;
    conv_param.padding_h = 0;
    conv_param.padding_w = 0;
    conv_param.use_bias = false;
    auto conv_op = std::make_shared<operators::Conv2D>(conv_param);
    conv_node->set_operator(conv_op);
    
    auto weight = std::make_shared<core::Tensor>(
        core::Shape({32, 3, 3, 3}),
        core::DataType::FLOAT32
    );
    conv_node->set_input_tensors({nullptr, weight});
    
    // ReLU node
    auto relu_node = graph->create_node("relu1");
    auto relu_op = std::make_shared<operators::ReLU>();
    relu_node->set_operator(relu_op);
    
    // Connect
    graph->connect("input", "conv1");
    graph->connect("conv1", "relu1");
    
    graph->set_inputs({"input"});
    graph->set_outputs({"relu1"});
    
    // Create optimization profile
    auto profile = std::make_shared<OptimizationProfile>();
    profile->set_shape_range("input",
        core::Shape({1, 3, 224, 224}),   // min
        core::Shape({4, 3, 384, 384}),   // opt
        core::Shape({8, 3, 512, 512})    // max
    );
    
    // Build engine with profile
    EngineConfig config;
    config.enable_dynamic_shapes = true;
    config.optimization_profile = profile;
    config.enable_profiling = true;
    
    Engine engine(config);
    auto status = engine.build(graph);
    
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    // Check that input shape was set to optimal
    auto updated_input = graph->get_node("input");
    ASSERT_NE(updated_input, nullptr);
    ASSERT_FALSE(updated_input->output_tensors().empty());
    ASSERT_NE(updated_input->output_tensors()[0], nullptr);
    
    auto shape = updated_input->output_tensors()[0]->shape();
    EXPECT_EQ(shape.to_string(), "[4, 3, 384, 384]");  // Should be optimal shape
}

TEST(EngineWithProfileTest, WithoutProfile) {
    // Test that engine still works without profile
    auto graph = std::make_shared<graph::Graph>();
    
    auto input_node = graph->create_node("input");
    auto input_tensor = std::make_shared<core::Tensor>(
        core::Shape({1, 3, 224, 224}),
        core::DataType::FLOAT32
    );
    input_node->set_output_tensors({input_tensor});
    
    graph->set_inputs({"input"});
    graph->set_outputs({"input"});
    
    // Build without profile
    EngineConfig config;
    config.enable_dynamic_shapes = false;  // Disabled
    
    Engine engine(config);
    auto status = engine.build(graph);
    
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    // Input shape should remain unchanged
    auto shape = input_node->output_tensors()[0]->shape();
    EXPECT_EQ(shape.to_string(), "[1, 3, 224, 224]");
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


