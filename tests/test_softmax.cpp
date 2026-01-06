#include <gtest/gtest.h>

#include <cmath>

#include "mini_infer/core/tensor.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/operators/plugin_base.h"

using namespace mini_infer;

class SoftmaxPluginTest : public ::testing::Test {
   protected:
    void SetUp() override {
        softmax_plugin_ = operators::PluginRegistry::instance().create_plugin(
            core::OpType::kSOFTMAX, core::DeviceType::CPU);
    }

    void TearDown() override {
        softmax_plugin_.reset();
    }

    std::unique_ptr<operators::IPlugin> softmax_plugin_;
};

TEST_F(SoftmaxPluginTest, BasicFunctionality) {
    ASSERT_NE(softmax_plugin_, nullptr);

    core::Shape shape({1, 3});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    float* input_data = static_cast<float*>(input->data());
    input_data[0] = 1.0f;
    input_data[1] = 2.0f;
    input_data[2] = 3.0f;

    // Infer output shape
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    auto status = softmax_plugin_->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);

    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    operators::PluginContext ctx;
    status = softmax_plugin_->enqueue(inputs, outputs, ctx);
    EXPECT_EQ(status, core::Status::SUCCESS);

    // Verify output
    float* output_data = static_cast<float*>(output->data());

    float sum = 0.0f;
    for (int i = 0; i < 3; ++i) {
        sum += output_data[i];
        EXPECT_GT(output_data[i], 0.0f);
        EXPECT_LT(output_data[i], 1.0f);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(SoftmaxPluginTest, TwoDimensional) {
    ASSERT_NE(softmax_plugin_, nullptr);

    core::Shape shape({2, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    float* input_data = static_cast<float*>(input->data());
    for (int i = 0; i < 8; ++i) {
        input_data[i] = static_cast<float>(i);
    }

    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    softmax_plugin_->infer_output_shapes(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    operators::PluginContext ctx;
    auto status = softmax_plugin_->enqueue(inputs, outputs, ctx);
    EXPECT_EQ(status, core::Status::SUCCESS);

    // Verify each row sums to 1
    float* output_data = static_cast<float*>(output->data());
    for (int row = 0; row < 2; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < 4; ++col) {
            sum += output_data[row * 4 + col];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
}

TEST_F(SoftmaxPluginTest, InferShape) {
    ASSERT_NE(softmax_plugin_, nullptr);

    std::vector<core::Shape> test_shapes = {
        core::Shape({10}),
        core::Shape({5, 10}),
        core::Shape({2, 3, 4}),
        core::Shape({1, 3, 224, 224})
    };

    for (const auto& input_shape : test_shapes) {
        std::vector<core::Shape> input_shapes = {input_shape};
        std::vector<core::Shape> output_shapes;

        auto status = softmax_plugin_->infer_output_shapes(input_shapes, output_shapes);
        EXPECT_EQ(status, core::Status::SUCCESS);
        ASSERT_EQ(output_shapes.size(), 1);

        // Output shape should match input shape
        EXPECT_EQ(output_shapes[0].ndim(), input_shape.ndim());
        for (size_t i = 0; i < input_shape.ndim(); ++i) {
            EXPECT_EQ(output_shapes[0][i], input_shape[i]);
        }
    }
}

TEST_F(SoftmaxPluginTest, PluginClone) {
    ASSERT_NE(softmax_plugin_, nullptr);

    auto cloned = softmax_plugin_->clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->get_op_type(), core::OpType::kSOFTMAX);
    EXPECT_EQ(cloned->get_device_type(), core::DeviceType::CPU);
}
