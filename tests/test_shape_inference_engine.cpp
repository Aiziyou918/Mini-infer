#include <gtest/gtest.h>
#include <vector>
#include "mini_infer/runtime/shape_inference_engine.h"
#include "mini_infer/graph/graph.h"
#include "mini_infer/operators/generic_operator.h"
#include "mini_infer/operators/plugin_base.h"

using namespace mini_infer;
using namespace mini_infer::runtime;

class ShapeInferenceEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple graph: input -> conv -> relu -> pool
        graph = std::make_shared<graph::Graph>();
        
        // Input
        auto input_node = graph->create_node("input");
        graph->set_inputs({"input"});
        
        // Conv2D
        auto conv_node = graph->create_node("conv1");
        auto conv_param = std::make_shared<operators::Conv2DParam>();
        conv_param->stride_h = 1;
        conv_param->stride_w = 1;
        conv_param->padding_h = 1;
        conv_param->padding_w = 1;
        conv_param->use_bias = false;
        auto conv_op = std::make_shared<operators::GenericOperator>("conv1", core::OpType::kCONVOLUTION);
        conv_op->set_plugin_param(conv_param);
        conv_node->set_operator(conv_op);

        auto weight = std::make_shared<core::Tensor>(
            core::Shape({64, 3, 3, 3}),
            core::DataType::FLOAT32
        );
        conv_node->set_input_tensors({nullptr, weight});

        // ReLU
        auto relu_node = graph->create_node("relu1");
        auto relu_op = std::make_shared<operators::GenericOperator>("relu1", core::OpType::kRELU);
        relu_node->set_operator(relu_op);

        // MaxPool
        auto pool_node = graph->create_node("pool1");
        auto pool_param = std::make_shared<operators::PoolingParam>();
        pool_param->type = operators::PoolingType::MAX;
        pool_param->kernel_h = 2;
        pool_param->kernel_w = 2;
        pool_param->stride_h = 2;
        pool_param->stride_w = 2;
        auto pool_op = std::make_shared<operators::GenericOperator>("pool1", core::OpType::kMAX_POOL);
        pool_op->set_plugin_param(pool_param);
        pool_node->set_operator(pool_op);
        
        // Connect
        graph->connect("input", "conv1");
        graph->connect("conv1", "relu1");
        graph->connect("relu1", "pool1");
        
        graph->set_outputs({"pool1"});

    }
    
    std::shared_ptr<graph::Graph> graph;
};

TEST_F(ShapeInferenceEngineTest, BasicInference) {
    ShapeInferenceEngine engine(graph);
    engine.set_verbose(true);
    
    // Infer with shape [1, 3, 224, 224]
    std::unordered_map<std::string, core::Shape> input_shapes;
    input_shapes["input"] = core::Shape({1, 3, 224, 224});
    
    auto status = engine.infer_shapes(input_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    // Check conv output: should be [1, 64, 224, 224] (same padding)
    auto conv_shape = engine.get_inferred_shape("conv1");
    ASSERT_NE(conv_shape, nullptr);
    EXPECT_EQ(conv_shape->to_string(), "[1, 64, 224, 224]");
    
    // Check relu output: should be same as conv
    auto relu_shape = engine.get_inferred_shape("relu1");
    ASSERT_NE(relu_shape, nullptr);
    EXPECT_EQ(relu_shape->to_string(), "[1, 64, 224, 224]");
    
    // Check pool output: should be [1, 64, 112, 112] (stride 2)
    auto pool_shape = engine.get_inferred_shape("pool1");
    ASSERT_NE(pool_shape, nullptr);
    EXPECT_EQ(pool_shape->to_string(), "[1, 64, 112, 112]");
}

TEST_F(ShapeInferenceEngineTest, DynamicBatchSize) {
    ShapeInferenceEngine engine(graph);
    
    // Test with batch size 4
    std::unordered_map<std::string, core::Shape> input_shapes1;
    input_shapes1["input"] = core::Shape({4, 3, 224, 224});
    
    auto status = engine.infer_shapes(input_shapes1);
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    auto pool_shape1 = engine.get_inferred_shape("pool1");
    ASSERT_NE(pool_shape1, nullptr);
    EXPECT_EQ(pool_shape1->to_string(), "[4, 64, 112, 112]");
    
    // Test with batch size 8 (different shape)
    std::unordered_map<std::string, core::Shape> input_shapes2;
    input_shapes2["input"] = core::Shape({8, 3, 224, 224});
    
    status = engine.infer_shapes(input_shapes2);
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    auto pool_shape2 = engine.get_inferred_shape("pool1");
    ASSERT_NE(pool_shape2, nullptr);
    EXPECT_EQ(pool_shape2->to_string(), "[8, 64, 112, 112]");
}

TEST_F(ShapeInferenceEngineTest, ShapesChanged) {
    ShapeInferenceEngine engine(graph);
    
    // First inference
    std::unordered_map<std::string, core::Shape> input_shapes1;
    input_shapes1["input"] = core::Shape({1, 3, 224, 224});
    engine.infer_shapes(input_shapes1);
    
    // Same shape - should not have changed
    EXPECT_FALSE(engine.shapes_changed(input_shapes1));
    
    // Different shape - should have changed
    std::unordered_map<std::string, core::Shape> input_shapes2;
    input_shapes2["input"] = core::Shape({4, 3, 224, 224});
    EXPECT_TRUE(engine.shapes_changed(input_shapes2));
    
    // After re-inference, should match
    engine.infer_shapes(input_shapes2);
    EXPECT_FALSE(engine.shapes_changed(input_shapes2));
}

TEST_F(ShapeInferenceEngineTest, DifferentResolutions) {
    ShapeInferenceEngine engine(graph);
    
    // Test with 384x384
    std::unordered_map<std::string, core::Shape> input_shapes;
    input_shapes["input"] = core::Shape({1, 3, 384, 384});
    
    auto status = engine.infer_shapes(input_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    // Conv: [1, 64, 384, 384]
    auto conv_shape = engine.get_inferred_shape("conv1");
    ASSERT_NE(conv_shape, nullptr);
    EXPECT_EQ(conv_shape->to_string(), "[1, 64, 384, 384]");
    
    // Pool: [1, 64, 192, 192]
    auto pool_shape = engine.get_inferred_shape("pool1");
    ASSERT_NE(pool_shape, nullptr);
    EXPECT_EQ(pool_shape->to_string(), "[1, 64, 192, 192]");
}

TEST_F(ShapeInferenceEngineTest, CacheClear) {
    ShapeInferenceEngine engine(graph);
    
    std::unordered_map<std::string, core::Shape> input_shapes;
    input_shapes["input"] = core::Shape({1, 3, 224, 224});
    
    engine.infer_shapes(input_shapes);
    
    // Should have results
    EXPECT_NE(engine.get_inferred_shape("conv1"), nullptr);
    
    // Clear cache
    engine.clear_cache();
    
    // After clearing, shapes_changed should return true (no cached input shapes)
    // This is because last_input_shapes_ is empty after clear
    EXPECT_TRUE(engine.shapes_changed(input_shapes));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
