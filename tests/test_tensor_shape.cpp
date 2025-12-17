#include <gtest/gtest.h>
#include "mini_infer/core/tensor.h"

using namespace mini_infer;

TEST(TensorShapeTest, CreateWithShape) {
    core::Shape shape({1, 3, 224, 224});
    auto tensor = std::make_shared<core::Tensor>(shape, core::DataType::FLOAT32);
    
    EXPECT_EQ(tensor->shape().ndim(), 4);
    EXPECT_EQ(tensor->shape()[0], 1);
    EXPECT_EQ(tensor->shape()[1], 3);
    EXPECT_EQ(tensor->shape()[2], 224);
    EXPECT_EQ(tensor->shape()[3], 224);
    EXPECT_EQ(tensor->shape().numel(), 150528);
}

TEST(TensorShapeTest, ReshapeWithSameNumel) {
    core::Shape shape({2, 3, 4, 5});
    auto tensor = std::make_shared<core::Tensor>(shape, core::DataType::FLOAT32);
    
    // Reshape to same number of elements
    core::Shape new_shape({6, 20});
    tensor->reshape(new_shape);
    
    EXPECT_EQ(tensor->shape().ndim(), 2);
    EXPECT_EQ(tensor->shape()[0], 6);
    EXPECT_EQ(tensor->shape()[1], 20);
}

TEST(TensorShapeTest, ReshapeEmptyTensorFails) {
    auto tensor = std::make_shared<core::Tensor>();
    
    // Empty tensor (numel=0) cannot be reshaped
    core::Shape new_shape({1, 3, 224, 224});
    tensor->reshape(new_shape);
    
    // Shape should still be empty
    EXPECT_EQ(tensor->shape().ndim(), 0);
    EXPECT_EQ(tensor->shape().numel(), 0);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

