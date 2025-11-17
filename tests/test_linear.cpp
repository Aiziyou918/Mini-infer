#include "mini_infer/operators/linear.h"
#include "mini_infer/operators/operator.h"
#include "mini_infer/core/tensor.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace mini_infer;

class LinearTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// ============================================================================
// Linear Basic Tests
// ============================================================================

TEST_F(LinearTest, BasicForward) {
    // Create Linear layer: 3 -> 2 with bias
    operators::LinearParam param(3, 2, true);
    auto linear = std::make_shared<operators::Linear>(param);
    
    // Input: [2, 3] (batch_size=2, in_features=3)
    core::Shape input_shape({2, 3});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);
    
    float* input_data = static_cast<float*>(input->data());
    // Batch 0: [1, 2, 3]
    input_data[0] = 1.0f;
    input_data[1] = 2.0f;
    input_data[2] = 3.0f;
    // Batch 1: [4, 5, 6]
    input_data[3] = 4.0f;
    input_data[4] = 5.0f;
    input_data[5] = 6.0f;
    
    // Weight: [2, 3] (out_features=2, in_features=3)
    core::Shape weight_shape({2, 3});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);
    ASSERT_NE(weight, nullptr);
    
    float* weight_data = static_cast<float*>(weight->data());
    // Output 0: [1, 0, -1]
    weight_data[0] = 1.0f;
    weight_data[1] = 0.0f;
    weight_data[2] = -1.0f;
    // Output 1: [0, 1, 0]
    weight_data[3] = 0.0f;
    weight_data[4] = 1.0f;
    weight_data[5] = 0.0f;
    
    // Bias: [2]
    core::Shape bias_shape({2});
    auto bias = core::Tensor::create(bias_shape, core::DataType::FLOAT32);
    ASSERT_NE(bias, nullptr);
    
    float* bias_data = static_cast<float*>(bias->data());
    bias_data[0] = 0.5f;
    bias_data[1] = -0.5f;
    
    // Forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight, bias};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = linear->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(outputs.size(), 1);
    
    // Verify output shape: [2, 2]
    auto output = outputs[0];
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->shape().ndim(), 2);
    EXPECT_EQ(output->shape()[0], 2);
    EXPECT_EQ(output->shape()[1], 2);
    
    float* output_data = static_cast<float*>(output->data());
    
    // Batch 0: [1, 2, 3] @ [[1, 0, -1], [0, 1, 0]]^T + [0.5, -0.5]
    // = [1*1 + 2*0 + 3*(-1), 1*0 + 2*1 + 3*0] + [0.5, -0.5]
    // = [-2, 2] + [0.5, -0.5] = [-1.5, 1.5]
    EXPECT_FLOAT_EQ(output_data[0], -1.5f);
    EXPECT_FLOAT_EQ(output_data[1], 1.5f);
    
    // Batch 1: [4, 5, 6] @ [[1, 0, -1], [0, 1, 0]]^T + [0.5, -0.5]
    // = [4*1 + 5*0 + 6*(-1), 4*0 + 5*1 + 6*0] + [0.5, -0.5]
    // = [-2, 5] + [0.5, -0.5] = [-1.5, 4.5]
    EXPECT_FLOAT_EQ(output_data[2], -1.5f);
    EXPECT_FLOAT_EQ(output_data[3], 4.5f);
}

TEST_F(LinearTest, WithoutBias) {
    // Create Linear layer: 2 -> 2 without bias
    operators::LinearParam param(2, 2, false);
    auto linear = std::make_shared<operators::Linear>(param);
    
    // Input: [1, 2]
    core::Shape input_shape({1, 2});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());
    input_data[0] = 3.0f;
    input_data[1] = 4.0f;
    
    // Weight: [2, 2] (identity matrix)
    core::Shape weight_shape({2, 2});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);
    float* weight_data = static_cast<float*>(weight->data());
    weight_data[0] = 1.0f;
    weight_data[1] = 0.0f;
    weight_data[2] = 0.0f;
    weight_data[3] = 1.0f;
    
    // Forward pass (no bias)
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = linear->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(outputs.size(), 1);
    
    float* output_data = static_cast<float*>(outputs[0]->data());
    
    // [3, 4] @ [[1, 0], [0, 1]]^T = [3, 4]
    EXPECT_FLOAT_EQ(output_data[0], 3.0f);
    EXPECT_FLOAT_EQ(output_data[1], 4.0f);
}

TEST_F(LinearTest, LargeBatch) {
    // Test with larger batch size
    operators::LinearParam param(4, 3, false);
    auto linear = std::make_shared<operators::Linear>(param);
    
    int batch_size = 10;
    int in_features = 4;
    int out_features = 3;
    
    // Input: [10, 4]
    core::Shape input_shape({batch_size, in_features});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());
    for (int i = 0; i < batch_size * in_features; ++i) {
        input_data[i] = static_cast<float>(i % 10);
    }
    
    // Weight: [3, 4]
    core::Shape weight_shape({out_features, in_features});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);
    float* weight_data = static_cast<float*>(weight->data());
    for (int i = 0; i < out_features * in_features; ++i) {
        weight_data[i] = 1.0f;
    }
    
    // Forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = linear->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(outputs.size(), 1);
    
    // Verify output shape
    EXPECT_EQ(outputs[0]->shape()[0], batch_size);
    EXPECT_EQ(outputs[0]->shape()[1], out_features);
}

// ============================================================================
// Shape Inference Tests
// ============================================================================

TEST_F(LinearTest, InferShapeBasic) {
    operators::LinearParam param(3, 2, true);
    auto linear = std::make_shared<operators::Linear>(param);
    
    std::vector<core::Shape> input_shapes = {
        core::Shape({5, 3}),  // input: [batch_size, in_features]
        core::Shape({2, 3}),  // weight: [out_features, in_features]
        core::Shape({2})      // bias: [out_features]
    };
    
    std::vector<core::Shape> output_shapes;
    auto status = linear->infer_shape(input_shapes, output_shapes);
    
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);
    
    // Output shape should be [5, 2]
    EXPECT_EQ(output_shapes[0].ndim(), 2);
    EXPECT_EQ(output_shapes[0][0], 5);
    EXPECT_EQ(output_shapes[0][1], 2);
}

TEST_F(LinearTest, InferShapeMultidimensional) {
    operators::LinearParam param(4, 3, false);
    auto linear = std::make_shared<operators::Linear>(param);
    
    std::vector<core::Shape> input_shapes = {
        core::Shape({2, 3, 4}),  // input: [2, 3, 4] - last dim is in_features
        core::Shape({3, 4})      // weight: [3, 4]
    };
    
    std::vector<core::Shape> output_shapes;
    auto status = linear->infer_shape(input_shapes, output_shapes);
    
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);
    
    // Output shape should be [2, 3, 3]
    EXPECT_EQ(output_shapes[0].ndim(), 3);
    EXPECT_EQ(output_shapes[0][0], 2);
    EXPECT_EQ(output_shapes[0][1], 3);
    EXPECT_EQ(output_shapes[0][2], 3);
}

// ============================================================================
// Operator Factory Tests
// ============================================================================

TEST_F(LinearTest, OperatorFactory) {
    auto linear = operators::OperatorFactory::create_operator("Linear");
    ASSERT_NE(linear, nullptr);
    EXPECT_EQ(linear->name(), "Linear");
}

// ============================================================================
// Data Type Tests
// ============================================================================

TEST_F(LinearTest, INT32DataType) {
    operators::LinearParam param(2, 2, false);
    auto linear = std::make_shared<operators::Linear>(param);
    
    // Input: [1, 2]
    core::Shape input_shape({1, 2});
    auto input = core::Tensor::create(input_shape, core::DataType::INT32);
    int32_t* input_data = static_cast<int32_t*>(input->data());
    input_data[0] = 3;
    input_data[1] = 4;
    
    // Weight: [2, 2]
    core::Shape weight_shape({2, 2});
    auto weight = core::Tensor::create(weight_shape, core::DataType::INT32);
    int32_t* weight_data = static_cast<int32_t*>(weight->data());
    weight_data[0] = 2;
    weight_data[1] = 0;
    weight_data[2] = 0;
    weight_data[3] = 3;
    
    // Forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = linear->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(outputs.size(), 1);
    
    int32_t* output_data = static_cast<int32_t*>(outputs[0]->data());
    
    // [3, 4] @ [[2, 0], [0, 3]]^T = [6, 12]
    EXPECT_EQ(output_data[0], 6);
    EXPECT_EQ(output_data[1], 12);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(LinearTest, SingleFeature) {
    // Test with single input/output features
    operators::LinearParam param(1, 1, false);
    auto linear = std::make_shared<operators::Linear>(param);
    
    core::Shape input_shape({3, 1});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());
    input_data[0] = 2.0f;
    input_data[1] = 3.0f;
    input_data[2] = 4.0f;
    
    core::Shape weight_shape({1, 1});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);
    float* weight_data = static_cast<float*>(weight->data());
    weight_data[0] = 5.0f;
    
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = linear->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    float* output_data = static_cast<float*>(outputs[0]->data());
    EXPECT_FLOAT_EQ(output_data[0], 10.0f);
    EXPECT_FLOAT_EQ(output_data[1], 15.0f);
    EXPECT_FLOAT_EQ(output_data[2], 20.0f);
}
