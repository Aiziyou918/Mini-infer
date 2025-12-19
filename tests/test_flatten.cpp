#include <gtest/gtest.h>

#include "mini_infer/core/tensor.h"
#include "mini_infer/operators/flatten.h"


using namespace mini_infer;

class FlattenTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(FlattenTest, BasicFlatten) {
    operators::FlattenParam param(1);
    auto flatten = std::make_shared<operators::Flatten>(param);

    core::Shape shape({2, 3, 4, 5});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    flatten->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = flatten->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(outputs.size(), 1);

    ASSERT_NE(outputs[0], nullptr);
    EXPECT_EQ(outputs[0]->shape().ndim(), 2);
    EXPECT_EQ(outputs[0]->shape()[0], 2);
    EXPECT_EQ(outputs[0]->shape()[1], 60);
}

TEST_F(FlattenTest, FlattenAxis0) {
    operators::FlattenParam param(0);
    auto flatten = std::make_shared<operators::Flatten>(param);

    core::Shape shape({2, 3, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    flatten->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = flatten->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape().ndim(), 2);
    EXPECT_EQ(outputs[0]->shape()[0], 1);
    EXPECT_EQ(outputs[0]->shape()[1], 24);
}

TEST_F(FlattenTest, FlattenAxis2) {
    operators::FlattenParam param(2);
    auto flatten = std::make_shared<operators::Flatten>(param);

    core::Shape shape({2, 3, 4, 5});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    flatten->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = flatten->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape().ndim(), 2);
    EXPECT_EQ(outputs[0]->shape()[0], 6);
    EXPECT_EQ(outputs[0]->shape()[1], 20);
}

TEST_F(FlattenTest, InferShape) {
    operators::FlattenParam param(1);
    auto flatten = std::make_shared<operators::Flatten>(param);

    std::vector<core::Shape> test_shapes = {
        core::Shape({1, 3, 224, 224}), core::Shape({2, 512, 7, 7}), core::Shape({4, 128, 14, 14})};

    std::vector<std::pair<int64_t, int64_t>> expected = {{1, 150528}, {2, 25088}, {4, 25088}};

    for (size_t i = 0; i < test_shapes.size(); ++i) {
        std::vector<core::Shape> input_shapes = {test_shapes[i]};
        std::vector<core::Shape> output_shapes;

        auto status = flatten->infer_shape(input_shapes, output_shapes);
        EXPECT_EQ(status, core::Status::SUCCESS);
        ASSERT_EQ(output_shapes.size(), 1);

        EXPECT_EQ(output_shapes[0].ndim(), 2);
        EXPECT_EQ(output_shapes[0][0], expected[i].first);
        EXPECT_EQ(output_shapes[0][1], expected[i].second);
    }
}

TEST_F(FlattenTest, DataPreservation) {
    operators::FlattenParam param(1);
    auto flatten = std::make_shared<operators::Flatten>(param);

    core::Shape shape({2, 3, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());

    for (int i = 0; i < 24; ++i) {
        input_data[i] = static_cast<float>(i);
    }

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    flatten->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = flatten->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    float* output_data = static_cast<float*>(outputs[0]->data());

    for (int i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(output_data[i], static_cast<float>(i));
    }
}

TEST_F(FlattenTest, TwoDimensionalInput) {
    operators::FlattenParam param(1);
    auto flatten = std::make_shared<operators::Flatten>(param);

    core::Shape shape({5, 10});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    flatten->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = flatten->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape().ndim(), 2);
    EXPECT_EQ(outputs[0]->shape()[0], 5);
    EXPECT_EQ(outputs[0]->shape()[1], 10);
}

TEST_F(FlattenTest, SingleBatch) {
    operators::FlattenParam param(1);
    auto flatten = std::make_shared<operators::Flatten>(param);

    core::Shape shape({1, 1000});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    flatten->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = flatten->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape()[0], 1);
    EXPECT_EQ(outputs[0]->shape()[1], 1000);
}

TEST_F(FlattenTest, SetParam) {
    auto flatten = std::make_shared<operators::Flatten>();

    operators::FlattenParam new_param(2);
    flatten->set_param(new_param);

    EXPECT_EQ(flatten->param().axis, 2);
}

TEST_F(FlattenTest, LargeInput) {
    operators::FlattenParam param(1);
    auto flatten = std::make_shared<operators::Flatten>(param);

    core::Shape shape({8, 512, 7, 7});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    flatten->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = flatten->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape()[0], 8);
    EXPECT_EQ(outputs[0]->shape()[1], 25088);
}
