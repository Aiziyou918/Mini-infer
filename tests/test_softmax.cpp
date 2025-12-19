#include <gtest/gtest.h>

#include <cmath>

#include "mini_infer/core/tensor.h"
#include "mini_infer/operators/softmax.h"


using namespace mini_infer;

class SoftmaxTest : public ::testing::Test {
   protected:
    void SetUp() override {
        softmax_ = std::make_shared<operators::Softmax>();
    }

    void TearDown() override {
        softmax_.reset();
    }

    std::shared_ptr<operators::Softmax> softmax_;
};

TEST_F(SoftmaxTest, BasicFunctionality) {
    core::Shape shape({1, 3});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    float* input_data = static_cast<float*>(input->data());
    input_data[0] = 1.0f;
    input_data[1] = 2.0f;
    input_data[2] = 3.0f;

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    softmax_->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = softmax_->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(outputs.size(), 1);

    // Verify output
    ASSERT_NE(outputs[0], nullptr);
    float* output_data = static_cast<float*>(outputs[0]->data());

    float sum = 0.0f;
    for (int i = 0; i < 3; ++i) {
        sum += output_data[i];
        EXPECT_GT(output_data[i], 0.0f);
        EXPECT_LT(output_data[i], 1.0f);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(SoftmaxTest, TwoDimensional) {
    core::Shape shape({2, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    float* input_data = static_cast<float*>(input->data());
    for (int i = 0; i < 8; ++i) {
        input_data[i] = static_cast<float>(i);
    }

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    softmax_->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = softmax_->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    float* output_data = static_cast<float*>(outputs[0]->data());

    for (int batch = 0; batch < 2; ++batch) {
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            int idx = batch * 4 + i;
            sum += output_data[idx];
            EXPECT_GT(output_data[idx], 0.0f);
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
}

TEST_F(SoftmaxTest, InferShape) {
    std::vector<core::Shape> test_shapes = {core::Shape({10}), core::Shape({5, 10}),
                                            core::Shape({2, 3, 4}), core::Shape({1, 1000})};

    for (const auto& input_shape : test_shapes) {
        std::vector<core::Shape> input_shapes = {input_shape};
        std::vector<core::Shape> output_shapes;

        auto status = softmax_->infer_shape(input_shapes, output_shapes);
        EXPECT_EQ(status, core::Status::SUCCESS);
        ASSERT_EQ(output_shapes.size(), 1);

        EXPECT_EQ(output_shapes[0].ndim(), input_shape.ndim());
        EXPECT_EQ(output_shapes[0].numel(), input_shape.numel());
        for (size_t i = 0; i < input_shape.ndim(); ++i) {
            EXPECT_EQ(output_shapes[0][i], input_shape[i]);
        }
    }
}

TEST_F(SoftmaxTest, AllSameValues) {
    core::Shape shape({1, 5});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());

    for (int i = 0; i < 5; ++i) {
        input_data[i] = 2.0f;
    }

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    softmax_->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = softmax_->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    float* output_data = static_cast<float*>(outputs[0]->data());
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(output_data[i], 0.2f, 1e-5f);
    }
}

TEST_F(SoftmaxTest, NegativeValues) {
    core::Shape shape({1, 3});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());

    input_data[0] = -1.0f;
    input_data[1] = -2.0f;
    input_data[2] = -3.0f;

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    softmax_->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = softmax_->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    float* output_data = static_cast<float*>(outputs[0]->data());
    float sum = 0.0f;
    for (int i = 0; i < 3; ++i) {
        sum += output_data[i];
        EXPECT_GT(output_data[i], 0.0f);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(SoftmaxTest, LargeValues) {
    core::Shape shape({1, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());

    input_data[0] = 100.0f;
    input_data[1] = 200.0f;
    input_data[2] = 300.0f;
    input_data[3] = 400.0f;

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    softmax_->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = softmax_->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    float* output_data = static_cast<float*>(outputs[0]->data());
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        sum += output_data[i];
        EXPECT_FALSE(std::isnan(output_data[i]));
        EXPECT_FALSE(std::isinf(output_data[i]));
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}
