#include <gtest/gtest.h>

#include "mini_infer/core/tensor.h"
#include "mini_infer/operators/reshape.h"


using namespace mini_infer;

class ReshapeTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ReshapeTest, BasicReshape) {
    operators::ReshapeParam param({2, 12});
    auto reshape = std::make_shared<operators::Reshape>(param);

    core::Shape shape({2, 3, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    reshape->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = reshape->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(outputs.size(), 1);

    ASSERT_NE(outputs[0], nullptr);
    EXPECT_EQ(outputs[0]->shape().ndim(), 2);
    EXPECT_EQ(outputs[0]->shape()[0], 2);
    EXPECT_EQ(outputs[0]->shape()[1], 12);
}

TEST_F(ReshapeTest, ReshapeWithInference) {
    operators::ReshapeParam param({2, -1});
    auto reshape = std::make_shared<operators::Reshape>(param);

    core::Shape shape({2, 3, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    reshape->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = reshape->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape()[0], 2);
    EXPECT_EQ(outputs[0]->shape()[1], 12);
}

TEST_F(ReshapeTest, ReshapeToVector) {
    operators::ReshapeParam param({-1});
    auto reshape = std::make_shared<operators::Reshape>(param);

    core::Shape shape({2, 3, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    reshape->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = reshape->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape().ndim(), 1);
    EXPECT_EQ(outputs[0]->shape()[0], 24);
}

TEST_F(ReshapeTest, ReshapeTo4D) {
    operators::ReshapeParam param({1, 3, 2, 4});
    auto reshape = std::make_shared<operators::Reshape>(param);

    core::Shape shape({6, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    reshape->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = reshape->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape().ndim(), 4);
    EXPECT_EQ(outputs[0]->shape()[0], 1);
    EXPECT_EQ(outputs[0]->shape()[1], 3);
    EXPECT_EQ(outputs[0]->shape()[2], 2);
    EXPECT_EQ(outputs[0]->shape()[3], 4);
}

TEST_F(ReshapeTest, InferShape) {
    operators::ReshapeParam param({4, -1});
    auto reshape = std::make_shared<operators::Reshape>(param);

    core::Shape input_shape({2, 3, 4});
    std::vector<core::Shape> input_shapes = {input_shape};
    std::vector<core::Shape> output_shapes;

    auto status = reshape->infer_shape(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);

    EXPECT_EQ(output_shapes[0].ndim(), 2);
    EXPECT_EQ(output_shapes[0][0], 4);
    EXPECT_EQ(output_shapes[0][1], 6);
}

TEST_F(ReshapeTest, DataPreservation) {
    operators::ReshapeParam param({4, 6});
    auto reshape = std::make_shared<operators::Reshape>(param);

    core::Shape shape({2, 3, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    float* input_data = static_cast<float*>(input->data());

    for (int i = 0; i < 24; ++i) {
        input_data[i] = static_cast<float>(i);
    }

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    reshape->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = reshape->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    float* output_data = static_cast<float*>(outputs[0]->data());

    for (int i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(output_data[i], static_cast<float>(i));
    }
}

TEST_F(ReshapeTest, MultipleInference) {
    operators::ReshapeParam param({2, -1, 3});
    auto reshape = std::make_shared<operators::Reshape>(param);

    core::Shape shape({6, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    reshape->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = reshape->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape()[0], 2);
    EXPECT_EQ(outputs[0]->shape()[1], 4);
    EXPECT_EQ(outputs[0]->shape()[2], 3);
}

TEST_F(ReshapeTest, SetParam) {
    auto reshape = std::make_shared<operators::Reshape>();

    operators::ReshapeParam new_param({3, -1});
    reshape->set_param(new_param);

    EXPECT_EQ(reshape->param().shape.size(), 2);
    EXPECT_EQ(reshape->param().shape[0], 3);
    EXPECT_EQ(reshape->param().shape[1], -1);
}

TEST_F(ReshapeTest, IdentityReshape) {
    operators::ReshapeParam param({2, 3, 4});
    auto reshape = std::make_shared<operators::Reshape>(param);

    core::Shape shape({2, 3, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    reshape->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = reshape->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape().ndim(), 3);
    EXPECT_EQ(outputs[0]->shape()[0], 2);
    EXPECT_EQ(outputs[0]->shape()[1], 3);
    EXPECT_EQ(outputs[0]->shape()[2], 4);
}

TEST_F(ReshapeTest, LargeReshape) {
    operators::ReshapeParam param({1, -1});
    auto reshape = std::make_shared<operators::Reshape>(param);

    core::Shape shape({8, 512, 7, 7});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    reshape->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = reshape->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape()[0], 1);
    EXPECT_EQ(outputs[0]->shape()[1], 200704);
}

TEST_F(ReshapeTest, BatchReshape) {
    operators::ReshapeParam param({-1, 12});
    auto reshape = std::make_shared<operators::Reshape>(param);

    core::Shape shape({4, 3, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    // Infer output shape and create output tensor
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    reshape->infer_shape(input_shapes, output_shapes);
    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    auto status = reshape->forward(inputs, outputs);
    EXPECT_EQ(status, core::Status::SUCCESS);

    EXPECT_EQ(outputs[0]->shape()[0], 4);
    EXPECT_EQ(outputs[0]->shape()[1], 12);
}
