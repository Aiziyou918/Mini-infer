#include <gtest/gtest.h>

#include "mini_infer/core/tensor.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/operators/plugin_base.h"
#include "mini_infer/operators/generic_operator.h"

using namespace mini_infer;

// ============================================================================
// Shape Utility Tests
// ============================================================================

TEST(ShapeTest, IsDynamic) {
    // Static shape
    core::Shape static_shape({1, 3, 224, 224});
    EXPECT_FALSE(static_shape.is_dynamic());

    // Dynamic shape with -1
    core::Shape dynamic_shape({-1, 3, 224, 224});
    EXPECT_TRUE(dynamic_shape.is_dynamic());

    // Dynamic shape with 0
    core::Shape zero_shape({0, 3, 224, 224});
    EXPECT_TRUE(zero_shape.is_dynamic());
}

TEST(ShapeTest, NumelDynamic) {
    core::Shape static_shape({2, 3, 4});
    EXPECT_EQ(static_shape.numel(), 24);

    core::Shape dynamic_shape({-1, 3, 4});
    EXPECT_EQ(dynamic_shape.numel(), -1);
}

// ============================================================================
// Plugin-based Shape Inference Tests
// ============================================================================

TEST(ShapeInferenceTest, ReLUInferShape) {
    auto relu_plugin = operators::PluginRegistry::instance().create_plugin(
        core::OpType::kRELU, core::DeviceType::CPU);
    ASSERT_NE(relu_plugin, nullptr);

    std::vector<core::Shape> test_shapes = {
        core::Shape({10}),
        core::Shape({5, 10}),
        core::Shape({2, 3, 4}),
        core::Shape({1, 3, 224, 224})
    };

    for (const auto& input_shape : test_shapes) {
        std::vector<core::Shape> input_shapes = {input_shape};
        std::vector<core::Shape> output_shapes;

        auto status = relu_plugin->infer_output_shapes(input_shapes, output_shapes);
        EXPECT_EQ(status, core::Status::SUCCESS);
        ASSERT_EQ(output_shapes.size(), 1);

        // ReLU should preserve input shape
        EXPECT_EQ(output_shapes[0].ndim(), input_shape.ndim());
        for (size_t i = 0; i < input_shape.ndim(); ++i) {
            EXPECT_EQ(output_shapes[0][i], input_shape[i]);
        }
    }
}

TEST(ShapeInferenceTest, FlattenInferShape) {
    auto flatten_plugin = operators::PluginRegistry::instance().create_plugin(
        core::OpType::kFLATTEN, core::DeviceType::CPU);
    ASSERT_NE(flatten_plugin, nullptr);

    // Test axis=1
    auto param = std::make_shared<operators::FlattenParam>(1);
    flatten_plugin->set_param(param);

    std::vector<core::Shape> input_shapes = {core::Shape({2, 3, 4, 5})};
    std::vector<core::Shape> output_shapes;

    auto status = flatten_plugin->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);

    EXPECT_EQ(output_shapes[0].ndim(), 2);
    EXPECT_EQ(output_shapes[0][0], 2);
    EXPECT_EQ(output_shapes[0][1], 60); // 3*4*5
}

TEST(ShapeInferenceTest, ReshapeInferShape) {
    auto reshape_plugin = operators::PluginRegistry::instance().create_plugin(
        core::OpType::kRESHAPE, core::DeviceType::CPU);
    ASSERT_NE(reshape_plugin, nullptr);

    // Test reshape to [2, 12]
    auto param = std::make_shared<operators::ReshapeParam>(std::vector<int64_t>{2, 12});
    reshape_plugin->set_param(param);

    std::vector<core::Shape> input_shapes = {core::Shape({2, 3, 4})};
    std::vector<core::Shape> output_shapes;

    auto status = reshape_plugin->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);

    EXPECT_EQ(output_shapes[0].ndim(), 2);
    EXPECT_EQ(output_shapes[0][0], 2);
    EXPECT_EQ(output_shapes[0][1], 12);
}

TEST(ShapeInferenceTest, Conv2DInferShape) {
    auto conv_plugin = operators::PluginRegistry::instance().create_plugin(
        core::OpType::kCONVOLUTION, core::DeviceType::CPU);
    ASSERT_NE(conv_plugin, nullptr);

    // 3x3 kernel, stride=1, padding=0
    auto param = std::make_shared<operators::Conv2DParam>(3, 3, 1, 1, 0, 0, 1, true);
    conv_plugin->set_param(param);

    std::vector<core::Shape> input_shapes = {
        core::Shape({1, 3, 224, 224}),   // input
        core::Shape({64, 3, 3, 3}),      // weight
        core::Shape({64})                 // bias
    };
    std::vector<core::Shape> output_shapes;

    auto status = conv_plugin->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);

    EXPECT_EQ(output_shapes[0][0], 1);
    EXPECT_EQ(output_shapes[0][1], 64);
    EXPECT_EQ(output_shapes[0][2], 222); // (224 - 3) / 1 + 1
    EXPECT_EQ(output_shapes[0][3], 222);
}

TEST(ShapeInferenceTest, PoolingInferShape) {
    auto pool_plugin = operators::PluginRegistry::instance().create_plugin(
        core::OpType::kMAX_POOL, core::DeviceType::CPU);
    ASSERT_NE(pool_plugin, nullptr);

    // 2x2 kernel, stride=2
    auto param = std::make_shared<operators::PoolingParam>(
        operators::PoolingType::MAX, 2, 2, 2, 2, 0, 0);
    pool_plugin->set_param(param);

    std::vector<core::Shape> input_shapes = {core::Shape({1, 64, 56, 56})};
    std::vector<core::Shape> output_shapes;

    auto status = pool_plugin->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);

    EXPECT_EQ(output_shapes[0][0], 1);
    EXPECT_EQ(output_shapes[0][1], 64);
    EXPECT_EQ(output_shapes[0][2], 28); // (56 - 2) / 2 + 1
    EXPECT_EQ(output_shapes[0][3], 28);
}

TEST(ShapeInferenceTest, LinearInferShape) {
    auto linear_plugin = operators::PluginRegistry::instance().create_plugin(
        core::OpType::kLINEAR, core::DeviceType::CPU);
    ASSERT_NE(linear_plugin, nullptr);

    auto param = std::make_shared<operators::LinearParam>(128, 10, true);
    linear_plugin->set_param(param);

    std::vector<core::Shape> input_shapes = {
        core::Shape({4, 128}),     // input
        core::Shape({10, 128}),    // weight
        core::Shape({10})          // bias
    };
    std::vector<core::Shape> output_shapes;

    auto status = linear_plugin->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);

    EXPECT_EQ(output_shapes[0][0], 4);
    EXPECT_EQ(output_shapes[0][1], 10);
}
