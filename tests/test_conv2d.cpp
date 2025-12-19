#include <gtest/gtest.h>

#include "mini_infer/core/tensor.h"
#include "mini_infer/operators/conv2d.h"
#include "mini_infer/operators/operator.h"

using namespace mini_infer;

class Conv2DTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// ============================================================================
// Conv2D Basic Tests
// ============================================================================

TEST_F(Conv2DTest, BasicConvolution) {
    // Create Conv2D: 3x3 kernel, stride=1, padding=0
    operators::Conv2DParam param(3, 3, 1, 1, 0, 0, 1, true);
    auto conv = std::make_shared<operators::Conv2D>(param);

    // Input: [1, 1, 5, 5]
    core::Shape input_shape({1, 1, 5, 5});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    float* input_data = static_cast<float*>(input->data());
    // Simple pattern
    for (int i = 0; i < 25; ++i) {
        input_data[i] = static_cast<float>(i + 1);
    }

    // Weight: [1, 1, 3, 3] (identity-like)
    core::Shape weight_shape({1, 1, 3, 3});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);
    ASSERT_NE(weight, nullptr);

    float* weight_data = static_cast<float*>(weight->data());
    // Simple averaging kernel
    for (int i = 0; i < 9; ++i) {
        weight_data[i] = 1.0f / 9.0f;
    }

    // Bias: [1]
    core::Shape bias_shape({1});
    auto bias = core::Tensor::create(bias_shape, core::DataType::FLOAT32);
    ASSERT_NE(bias, nullptr);

    float* bias_data = static_cast<float*>(bias->data());
    bias_data[0] = 0.0f;

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {input_shape, weight_shape, bias_shape};
    std::vector<core::Shape> output_shapes;
    conv->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    // Forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight, bias};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = conv->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(outputs.size(), 1);

    // Verify output shape: [1, 1, 3, 3]
    auto output_tensor = outputs[0];
    ASSERT_NE(output_tensor, nullptr);
    EXPECT_EQ(output_tensor->shape().ndim(), 4);
    EXPECT_EQ(output_tensor->shape()[0], 1);
    EXPECT_EQ(output_tensor->shape()[1], 1);
    EXPECT_EQ(output_tensor->shape()[2], 3);
    EXPECT_EQ(output_tensor->shape()[3], 3);

    // Output should be averaged values
    float* output_data = static_cast<float*>(output_tensor->data());

    // First element: average of [1,2,3,6,7,8,11,12,13]
    float expected_0 = (1 + 2 + 3 + 6 + 7 + 8 + 11 + 12 + 13) / 9.0f;
    EXPECT_NEAR(output_data[0], expected_0, 1e-5f);
}

TEST_F(Conv2DTest, ConvolutionWithPadding) {
    // Create Conv2D: 3x3 kernel, stride=1, padding=1
    operators::Conv2DParam param(3, 3, 1, 1, 1, 1, 1, false);
    auto conv = std::make_shared<operators::Conv2D>(param);

    // Input: [1, 1, 3, 3]
    core::Shape input_shape({1, 1, 3, 3});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);

    float* input_data = static_cast<float*>(input->data());
    for (int i = 0; i < 9; ++i) {
        input_data[i] = 1.0f;
    }

    // Weight: [1, 1, 3, 3] (all ones)
    core::Shape weight_shape({1, 1, 3, 3});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);

    float* weight_data = static_cast<float*>(weight->data());
    for (int i = 0; i < 9; ++i) {
        weight_data[i] = 1.0f;
    }

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {input_shape, weight_shape};
    std::vector<core::Shape> output_shapes;
    conv->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    // Forward pass (no bias)
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = conv->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(outputs.size(), 1);

    // With padding=1, output shape should be [1, 1, 3, 3]
    EXPECT_EQ(outputs[0]->shape()[2], 3);
    EXPECT_EQ(outputs[0]->shape()[3], 3);

    float* output_data = static_cast<float*>(outputs[0]->data());

    // Center pixel sees all 9 input pixels
    EXPECT_FLOAT_EQ(output_data[4], 9.0f);  // center position

    // Corner pixel sees 4 input pixels
    EXPECT_FLOAT_EQ(output_data[0], 4.0f);  // top-left corner
}

TEST_F(Conv2DTest, ConvolutionWithStride) {
    // Create Conv2D: 3x3 kernel, stride=2, padding=0
    operators::Conv2DParam param(3, 3, 2, 2, 0, 0, 1, false);
    auto conv = std::make_shared<operators::Conv2D>(param);

    // Input: [1, 1, 5, 5]
    core::Shape input_shape({1, 1, 5, 5});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);

    float* input_data = static_cast<float*>(input->data());
    for (int i = 0; i < 25; ++i) {
        input_data[i] = 1.0f;
    }

    // Weight: [1, 1, 3, 3]
    core::Shape weight_shape({1, 1, 3, 3});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);

    float* weight_data = static_cast<float*>(weight->data());
    for (int i = 0; i < 9; ++i) {
        weight_data[i] = 1.0f;
    }

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {input_shape, weight_shape};
    std::vector<core::Shape> output_shapes;
    conv->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    // Forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = conv->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    // With stride=2, output shape should be [1, 1, 2, 2]
    EXPECT_EQ(outputs[0]->shape()[2], 2);
    EXPECT_EQ(outputs[0]->shape()[3], 2);

    float* output_data = static_cast<float*>(outputs[0]->data());
    // Each output pixel should be sum of 9 input pixels
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(output_data[i], 9.0f);
    }
}

TEST_F(Conv2DTest, MultiChannelConvolution) {
    // Create Conv2D: 3x3 kernel, stride=1, padding=1
    operators::Conv2DParam param(3, 3, 1, 1, 1, 1, 1, false);
    auto conv = std::make_shared<operators::Conv2D>(param);

    // Input: [1, 2, 3, 3] (2 input channels)
    core::Shape input_shape({1, 2, 3, 3});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);

    float* input_data = static_cast<float*>(input->data());
    // Channel 0: all ones
    for (int i = 0; i < 9; ++i) {
        input_data[i] = 1.0f;
    }
    // Channel 1: all twos
    for (int i = 9; i < 18; ++i) {
        input_data[i] = 2.0f;
    }

    // Weight: [3, 2, 3, 3] (3 output channels, 2 input channels)
    core::Shape weight_shape({3, 2, 3, 3});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);

    float* weight_data = static_cast<float*>(weight->data());
    for (int i = 0; i < 54; ++i) {
        weight_data[i] = 0.1f;
    }

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {input_shape, weight_shape};
    std::vector<core::Shape> output_shapes;
    conv->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    // Forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = conv->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    // Output shape should be [1, 3, 3, 3]
    EXPECT_EQ(outputs[0]->shape()[0], 1);
    EXPECT_EQ(outputs[0]->shape()[1], 3);
    EXPECT_EQ(outputs[0]->shape()[2], 3);
    EXPECT_EQ(outputs[0]->shape()[3], 3);
}

TEST_F(Conv2DTest, BatchProcessing) {
    // Test with batch size > 1
    operators::Conv2DParam param(3, 3, 1, 1, 1, 1, 1, false);
    auto conv = std::make_shared<operators::Conv2D>(param);

    // Input: [2, 1, 3, 3] (batch size = 2)
    core::Shape input_shape({2, 1, 3, 3});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);

    float* input_data = static_cast<float*>(input->data());
    // Batch 0
    for (int i = 0; i < 9; ++i) {
        input_data[i] = 1.0f;
    }
    // Batch 1
    for (int i = 9; i < 18; ++i) {
        input_data[i] = 2.0f;
    }

    // Weight: [1, 1, 3, 3]
    core::Shape weight_shape({1, 1, 3, 3});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);

    float* weight_data = static_cast<float*>(weight->data());
    for (int i = 0; i < 9; ++i) {
        weight_data[i] = 1.0f;
    }

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {input_shape, weight_shape};
    std::vector<core::Shape> output_shapes;
    conv->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    // Forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = conv->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    // Output shape should be [2, 1, 3, 3]
    EXPECT_EQ(outputs[0]->shape()[0], 2);

    float* output_data = static_cast<float*>(outputs[0]->data());

    // Batch 0 center pixel
    EXPECT_FLOAT_EQ(output_data[4], 9.0f);

    // Batch 1 center pixel (input was 2.0)
    EXPECT_FLOAT_EQ(output_data[13], 18.0f);
}

// ============================================================================
// Shape Inference Tests
// ============================================================================

TEST_F(Conv2DTest, InferShapeBasic) {
    operators::Conv2DParam param(3, 3, 1, 1, 0, 0, 1, false);
    auto conv = std::make_shared<operators::Conv2D>(param);

    std::vector<core::Shape> input_shapes = {
        core::Shape({1, 3, 224, 224}),  // input
        core::Shape({64, 3, 3, 3})      // weight
    };

    std::vector<core::Shape> output_shapes;
    auto status = conv->infer_shape(input_shapes, output_shapes);

    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);

    // Output shape: [1, 64, 222, 222]
    EXPECT_EQ(output_shapes[0][0], 1);
    EXPECT_EQ(output_shapes[0][1], 64);
    EXPECT_EQ(output_shapes[0][2], 222);
    EXPECT_EQ(output_shapes[0][3], 222);
}

TEST_F(Conv2DTest, InferShapeWithPaddingStride) {
    operators::Conv2DParam param(3, 3, 2, 2, 1, 1, 1, true);
    auto conv = std::make_shared<operators::Conv2D>(param);

    std::vector<core::Shape> input_shapes = {
        core::Shape({2, 3, 32, 32}),  // input
        core::Shape({16, 3, 3, 3}),   // weight
        core::Shape({16})             // bias
    };

    std::vector<core::Shape> output_shapes;
    auto status = conv->infer_shape(input_shapes, output_shapes);

    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);

    // With padding=1, stride=2: (32+2-3)/2+1 = 16
    EXPECT_EQ(output_shapes[0][0], 2);
    EXPECT_EQ(output_shapes[0][1], 16);
    EXPECT_EQ(output_shapes[0][2], 16);
    EXPECT_EQ(output_shapes[0][3], 16);
}

// ============================================================================
// Operator Factory Tests
// ============================================================================

TEST_F(Conv2DTest, OperatorFactory) {
    auto conv = operators::OperatorFactory::create_operator("Conv");
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->name(), "Conv");
}
