#include <gtest/gtest.h>

#include "mini_infer/core/tensor.h"
#include "mini_infer/operators/pooling.h"


using namespace mini_infer;

class PoolingTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(PoolingTest, MaxPoolBasic) {
    operators::PoolingParam param(operators::PoolingType::MAX, 2, 2, 2, 2, 0, 0);
    auto pooling = std::make_shared<operators::Pooling>(param);

    core::Shape shape({1, 1, 4, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    float* input_data = static_cast<float*>(input->data());
    for (int i = 0; i < 16; ++i) {
        input_data[i] = static_cast<float>(i);
    }

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    pooling->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = pooling->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(outputs.size(), 1);

    ASSERT_NE(outputs[0], nullptr);
    EXPECT_EQ(outputs[0]->shape().ndim(), 4);
    EXPECT_EQ(outputs[0]->shape()[0], 1);
    EXPECT_EQ(outputs[0]->shape()[1], 1);
    EXPECT_EQ(outputs[0]->shape()[2], 2);
    EXPECT_EQ(outputs[0]->shape()[3], 2);

    float* output_data = static_cast<float*>(outputs[0]->data());
    EXPECT_FLOAT_EQ(output_data[0], 5.0f);
    EXPECT_FLOAT_EQ(output_data[1], 7.0f);
    EXPECT_FLOAT_EQ(output_data[2], 13.0f);
    EXPECT_FLOAT_EQ(output_data[3], 15.0f);
}

TEST_F(PoolingTest, AvgPoolBasic) {
    operators::PoolingParam param(operators::PoolingType::AVERAGE, 2, 2, 2, 2, 0, 0);
    auto pooling = std::make_shared<operators::Pooling>(param);

    core::Shape shape({1, 1, 4, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    float* input_data = static_cast<float*>(input->data());
    for (int i = 0; i < 16; ++i) {
        input_data[i] = static_cast<float>(i);
    }

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    pooling->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = pooling->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(outputs.size(), 1);

    float* output_data = static_cast<float*>(outputs[0]->data());

    EXPECT_FLOAT_EQ(output_data[0], 2.5f);
    EXPECT_FLOAT_EQ(output_data[1], 4.5f);
    EXPECT_FLOAT_EQ(output_data[2], 10.5f);
    EXPECT_FLOAT_EQ(output_data[3], 12.5f);
}

TEST_F(PoolingTest, MaxPoolWithPadding) {
    operators::PoolingParam param(operators::PoolingType::MAX, 2, 2, 2, 2, 1, 1);
    auto pooling = std::make_shared<operators::Pooling>(param);

    core::Shape shape({1, 1, 3, 3});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());

    for (int i = 0; i < 9; ++i) {
        input_data[i] = static_cast<float>(i + 1);
    }

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    pooling->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = pooling->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape()[2], 2);
    EXPECT_EQ(outputs[0]->shape()[3], 2);
}

TEST_F(PoolingTest, InferShapeMaxPool) {
    operators::PoolingParam param(operators::PoolingType::MAX, 2, 2, 2, 2, 0, 0);
    auto pooling = std::make_shared<operators::Pooling>(param);

    core::Shape input_shape({1, 3, 224, 224});
    std::vector<core::Shape> input_shapes = {input_shape};
    std::vector<core::Shape> output_shapes;

    auto status = pooling->infer_shape(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);

    EXPECT_EQ(output_shapes[0][0], 1);
    EXPECT_EQ(output_shapes[0][1], 3);
    EXPECT_EQ(output_shapes[0][2], 112);
    EXPECT_EQ(output_shapes[0][3], 112);
}

TEST_F(PoolingTest, InferShapeWithPadding) {
    operators::PoolingParam param(operators::PoolingType::MAX, 3, 3, 2, 2, 1, 1);
    auto pooling = std::make_shared<operators::Pooling>(param);

    core::Shape input_shape({1, 64, 56, 56});
    std::vector<core::Shape> input_shapes = {input_shape};
    std::vector<core::Shape> output_shapes;

    auto status = pooling->infer_shape(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);

    EXPECT_EQ(output_shapes[0][0], 1);
    EXPECT_EQ(output_shapes[0][1], 64);
    EXPECT_EQ(output_shapes[0][2], 28);
    EXPECT_EQ(output_shapes[0][3], 28);
}

TEST_F(PoolingTest, MultiChannelMaxPool) {
    operators::PoolingParam param(operators::PoolingType::MAX, 2, 2, 2, 2, 0, 0);
    auto pooling = std::make_shared<operators::Pooling>(param);

    core::Shape shape({1, 2, 4, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());

    for (int i = 0; i < 32; ++i) {
        input_data[i] = static_cast<float>(i);
    }

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    pooling->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = pooling->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape()[0], 1);
    EXPECT_EQ(outputs[0]->shape()[1], 2);
    EXPECT_EQ(outputs[0]->shape()[2], 2);
    EXPECT_EQ(outputs[0]->shape()[3], 2);
}

TEST_F(PoolingTest, AsymmetricKernel) {
    operators::PoolingParam param(operators::PoolingType::MAX, 3, 2, 3, 2, 0, 0);
    auto pooling = std::make_shared<operators::Pooling>(param);

    core::Shape input_shape({1, 1, 6, 6});
    std::vector<core::Shape> input_shapes = {input_shape};
    std::vector<core::Shape> output_shapes;

    auto status = pooling->infer_shape(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(output_shapes[0][2], 2);
    EXPECT_EQ(output_shapes[0][3], 3);
}

TEST_F(PoolingTest, AvgPoolMultiChannel) {
    operators::PoolingParam param(operators::PoolingType::AVERAGE, 2, 2, 2, 2, 0, 0);
    auto pooling = std::make_shared<operators::Pooling>(param);

    core::Shape shape({2, 3, 4, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());

    for (int i = 0; i < 2 * 3 * 4 * 4; ++i) {
        input_data[i] = 1.0f;
    }

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    pooling->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = pooling->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape()[0], 2);
    EXPECT_EQ(outputs[0]->shape()[1], 3);
    EXPECT_EQ(outputs[0]->shape()[2], 2);
    EXPECT_EQ(outputs[0]->shape()[3], 2);

    float* output_data = static_cast<float*>(outputs[0]->data());
    for (int i = 0; i < 2 * 3 * 2 * 2; ++i) {
        EXPECT_FLOAT_EQ(output_data[i], 1.0f);
    }
}
