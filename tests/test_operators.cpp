#include "mini_infer/operators/relu.h"
#include "mini_infer/operators/operator.h"
#include "mini_infer/core/tensor.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace mini_infer;

class OperatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        relu_ = std::make_shared<operators::ReLU>();
    }

    void TearDown() override {
        relu_.reset();
    }

    std::shared_ptr<operators::ReLU> relu_;
};

// ============================================================================
// ReLU Basic Tests
// ============================================================================

TEST_F(OperatorTest, ReLUBasicFunctionality) {
    core::Shape shape({1, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);
    
    // Fill test data: [-2.0, -1.0, 0.0, 1.0]
    float* input_data = static_cast<float*>(input->data());
    input_data[0] = -2.0f;
    input_data[1] = -1.0f;
    input_data[2] = 0.0f;
    input_data[3] = 1.0f;
    
    // Execute forward pass
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = relu_->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(outputs.size(), 1);
    
    // Verify output
    auto output = outputs[0];
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->shape().ndim(), shape.ndim());
    EXPECT_EQ(output->shape().numel(), shape.numel());
    
    float* output_data = static_cast<float*>(output->data());
    
    // Expected output: [0.0, 0.0, 0.0, 1.0]
    EXPECT_FLOAT_EQ(output_data[0], 0.0f);
    EXPECT_FLOAT_EQ(output_data[1], 0.0f);
    EXPECT_FLOAT_EQ(output_data[2], 0.0f);
    EXPECT_FLOAT_EQ(output_data[3], 1.0f);
}

TEST_F(OperatorTest, ReLUMultidimensional) {
    // Create 2D tensor [2, 3]
    core::Shape shape({2, 3});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);
    
    float* input_data = static_cast<float*>(input->data());
    input_data[0] = -3.0f;
    input_data[1] = 2.0f;
    input_data[2] = -1.0f;
    input_data[3] = 0.5f;
    input_data[4] = -0.5f;
    input_data[5] = 4.0f;
    
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = relu_->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    auto output = outputs[0];
    float* output_data = static_cast<float*>(output->data());
    
    // Expected output: [0.0, 2.0, 0.0, 0.5, 0.0, 4.0]
    EXPECT_FLOAT_EQ(output_data[0], 0.0f);
    EXPECT_FLOAT_EQ(output_data[1], 2.0f);
    EXPECT_FLOAT_EQ(output_data[2], 0.0f);
    EXPECT_FLOAT_EQ(output_data[3], 0.5f);
    EXPECT_FLOAT_EQ(output_data[4], 0.0f);
    EXPECT_FLOAT_EQ(output_data[5], 4.0f);
}

TEST_F(OperatorTest, ReLUInferShape) {
    std::vector<core::Shape> test_shapes = {
        core::Shape({10}),           // 1D
        core::Shape({5, 10}),        // 2D
        core::Shape({2, 3, 4}),      // 3D
        core::Shape({1, 3, 224, 224}) // 4D (image)
    };
    
    for (const auto& input_shape : test_shapes) {
        std::vector<core::Shape> input_shapes = {input_shape};
        std::vector<core::Shape> output_shapes;
        
        auto status = relu_->infer_shape(input_shapes, output_shapes);
        EXPECT_EQ(status, core::Status::SUCCESS);
        ASSERT_EQ(output_shapes.size(), 1);
        
        // ReLU doesn't change shape
        EXPECT_EQ(output_shapes[0].ndim(), input_shape.ndim());
        EXPECT_EQ(output_shapes[0].numel(), input_shape.numel());
        for (size_t i = 0; i < input_shape.ndim(); ++i) {
            EXPECT_EQ(output_shapes[0][i], input_shape[i]);
        }
    }
}

TEST_F(OperatorTest, ReLUOperatorFactory) {
    // Create ReLU through factory
    auto relu = operators::OperatorFactory::create_operator("ReLU");
    ASSERT_NE(relu, nullptr);
    EXPECT_EQ(relu->name(), "ReLU");
    
    // Test created operator
    core::Shape shape({1, 5});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());
    for (int i = 0; i < 5; ++i) {
        input_data[i] = static_cast<float>(i - 2); // [-2, -1, 0, 1, 2]
    }
    
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = relu->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(outputs.size(), 1);
    
    float* output_data = static_cast<float*>(outputs[0]->data());
    EXPECT_FLOAT_EQ(output_data[0], 0.0f);
    EXPECT_FLOAT_EQ(output_data[1], 0.0f);
    EXPECT_FLOAT_EQ(output_data[2], 0.0f);
    EXPECT_FLOAT_EQ(output_data[3], 1.0f);
    EXPECT_FLOAT_EQ(output_data[4], 2.0f);
}

TEST_F(OperatorTest, ReLUAllPositive) {
    core::Shape shape({3});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());
    input_data[0] = 1.0f;
    input_data[1] = 2.0f;
    input_data[2] = 3.0f;
    
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = relu_->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    float* output_data = static_cast<float*>(outputs[0]->data());
    EXPECT_FLOAT_EQ(output_data[0], 1.0f);
    EXPECT_FLOAT_EQ(output_data[1], 2.0f);
    EXPECT_FLOAT_EQ(output_data[2], 3.0f);
}

TEST_F(OperatorTest, ReLUAllNegative) {
    core::Shape shape({3});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());
    input_data[0] = -1.0f;
    input_data[1] = -2.0f;
    input_data[2] = -3.0f;
    
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = relu_->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    float* output_data = static_cast<float*>(outputs[0]->data());
    EXPECT_FLOAT_EQ(output_data[0], 0.0f);
    EXPECT_FLOAT_EQ(output_data[1], 0.0f);
    EXPECT_FLOAT_EQ(output_data[2], 0.0f);
}

TEST_F(OperatorTest, ReLUAllZero) {
    core::Shape shape({3});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());
    input_data[0] = 0.0f;
    input_data[1] = 0.0f;
    input_data[2] = 0.0f;
    
    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs;
    
    auto status = relu_->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    float* output_data = static_cast<float*>(outputs[0]->data());
    EXPECT_FLOAT_EQ(output_data[0], 0.0f);
    EXPECT_FLOAT_EQ(output_data[1], 0.0f);
    EXPECT_FLOAT_EQ(output_data[2], 0.0f);
}
