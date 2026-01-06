#include <gtest/gtest.h>

#include "mini_infer/core/tensor.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/operators/plugin_base.h"

using namespace mini_infer;

class Conv2DPluginTest : public ::testing::Test {
   protected:
    void SetUp() override {
        conv_plugin_ = operators::PluginRegistry::instance().create_plugin(
            core::OpType::kCONVOLUTION, core::DeviceType::CPU);
    }

    void TearDown() override {
        conv_plugin_.reset();
    }

    std::unique_ptr<operators::IPlugin> conv_plugin_;
};

// ============================================================================
// Conv2D Plugin Tests
// ============================================================================

TEST_F(Conv2DPluginTest, BasicConvolution) {
    ASSERT_NE(conv_plugin_, nullptr);

    // Create Conv2D: 3x3 kernel, stride=1, padding=0
    auto param = std::make_shared<operators::Conv2DParam>(3, 3, 1, 1, 0, 0, 1, true);
    conv_plugin_->set_param(param);

    // Input: [1, 1, 5, 5]
    core::Shape input_shape({1, 1, 5, 5});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    float* input_data = static_cast<float*>(input->data());
    for (int i = 0; i < 25; ++i) {
        input_data[i] = static_cast<float>(i + 1);
    }

    // Weight: [1, 1, 3, 3] (averaging kernel)
    core::Shape weight_shape({1, 1, 3, 3});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);
    ASSERT_NE(weight, nullptr);

    float* weight_data = static_cast<float*>(weight->data());
    for (int i = 0; i < 9; ++i) {
        weight_data[i] = 1.0f / 9.0f;
    }

    // Bias: [1]
    core::Shape bias_shape({1});
    auto bias = core::Tensor::create(bias_shape, core::DataType::FLOAT32);
    ASSERT_NE(bias, nullptr);

    float* bias_data = static_cast<float*>(bias->data());
    bias_data[0] = 0.0f;

    // Infer output shape
    std::vector<core::Shape> input_shapes = {input_shape, weight_shape, bias_shape};
    std::vector<core::Shape> output_shapes;
    auto status = conv_plugin_->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);

    // Output should be [1, 1, 3, 3] for 5x5 input with 3x3 kernel, stride 1, no padding
    EXPECT_EQ(output_shapes[0][0], 1);
    EXPECT_EQ(output_shapes[0][1], 1);
    EXPECT_EQ(output_shapes[0][2], 3);
    EXPECT_EQ(output_shapes[0][3], 3);

    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight, bias};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    operators::PluginContext ctx;
    status = conv_plugin_->enqueue(inputs, outputs, ctx);
    EXPECT_EQ(status, core::Status::SUCCESS);

    float* output_data = static_cast<float*>(output->data());
    // First element should be average of top-left 3x3 region
    float expected_first = (1 + 2 + 3 + 6 + 7 + 8 + 11 + 12 + 13) / 9.0f;
    EXPECT_NEAR(output_data[0], expected_first, 1e-5f);
}

TEST_F(Conv2DPluginTest, InferShapeWithPadding) {
    ASSERT_NE(conv_plugin_, nullptr);

    // 3x3 kernel, stride=1, padding=1 (same padding)
    auto param = std::make_shared<operators::Conv2DParam>(3, 3, 1, 1, 1, 1, 1, true);
    conv_plugin_->set_param(param);

    std::vector<core::Shape> input_shapes = {
        core::Shape({1, 3, 224, 224}),   // input
        core::Shape({64, 3, 3, 3}),      // weight
        core::Shape({64})                 // bias
    };
    std::vector<core::Shape> output_shapes;

    auto status = conv_plugin_->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);

    EXPECT_EQ(output_shapes[0][0], 1);
    EXPECT_EQ(output_shapes[0][1], 64);
    EXPECT_EQ(output_shapes[0][2], 224);
    EXPECT_EQ(output_shapes[0][3], 224);
}

TEST_F(Conv2DPluginTest, InferShapeWithStride) {
    ASSERT_NE(conv_plugin_, nullptr);

    // 3x3 kernel, stride=2, padding=1
    auto param = std::make_shared<operators::Conv2DParam>(3, 3, 2, 2, 1, 1, 1, true);
    conv_plugin_->set_param(param);

    std::vector<core::Shape> input_shapes = {
        core::Shape({1, 3, 224, 224}),
        core::Shape({64, 3, 3, 3}),
        core::Shape({64})
    };
    std::vector<core::Shape> output_shapes;

    auto status = conv_plugin_->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);

    EXPECT_EQ(output_shapes[0][0], 1);
    EXPECT_EQ(output_shapes[0][1], 64);
    EXPECT_EQ(output_shapes[0][2], 112);
    EXPECT_EQ(output_shapes[0][3], 112);
}

TEST_F(Conv2DPluginTest, PluginClone) {
    ASSERT_NE(conv_plugin_, nullptr);

    auto cloned = conv_plugin_->clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->get_op_type(), core::OpType::kCONVOLUTION);
    EXPECT_EQ(cloned->get_device_type(), core::DeviceType::CPU);
}
