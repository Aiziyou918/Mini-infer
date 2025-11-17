#include "mini_infer/core/tensor.h"
#include <gtest/gtest.h>

using namespace mini_infer;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// ============================================================================
// Shape Tests
// ============================================================================

TEST_F(TensorTest, ShapeCreation) {
    core::Shape shape({2, 3, 4, 5});
    
    EXPECT_EQ(shape.ndim(), 4);
    EXPECT_EQ(shape.numel(), 120);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_EQ(shape[3], 5);
}

TEST_F(TensorTest, ShapeToString) {
    core::Shape shape({2, 3, 4});
    std::string str = shape.to_string();
    
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("2"), std::string::npos);
    EXPECT_NE(str.find("3"), std::string::npos);
    EXPECT_NE(str.find("4"), std::string::npos);
}

TEST_F(TensorTest, ShapeEmptyNumel) {
    core::Shape empty_shape;
    EXPECT_EQ(empty_shape.ndim(), 0);
    EXPECT_EQ(empty_shape.numel(), 0);
}

// ============================================================================
// Tensor Creation Tests
// ============================================================================

TEST_F(TensorTest, TensorCreationFloat32) {
    core::Shape shape({2, 3, 224, 224});
    auto tensor = core::Tensor::create(shape, core::DataType::FLOAT32);
    
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape().ndim(), 4);
    EXPECT_EQ(tensor->shape()[0], 2);
    EXPECT_EQ(tensor->shape()[1], 3);
    EXPECT_EQ(tensor->shape()[2], 224);
    EXPECT_EQ(tensor->shape()[3], 224);
    EXPECT_FALSE(tensor->empty());
    EXPECT_EQ(tensor->dtype(), core::DataType::FLOAT32);
}

TEST_F(TensorTest, TensorCreationInt32) {
    core::Shape shape({10, 20});
    auto tensor = core::Tensor::create(shape, core::DataType::INT32);
    
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dtype(), core::DataType::INT32);
    EXPECT_EQ(tensor->shape().numel(), 200);
}

TEST_F(TensorTest, TensorDataAccess) {
    core::Shape shape({3, 3});
    auto tensor = core::Tensor::create(shape, core::DataType::FLOAT32);
    
    ASSERT_NE(tensor, nullptr);
    EXPECT_NE(tensor->data(), nullptr);
    
    // Write and read data
    float* data = static_cast<float*>(tensor->data());
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
}

// ============================================================================
// Tensor Reshape Tests
// ============================================================================

TEST_F(TensorTest, TensorReshapeValid) {
    core::Shape shape({2, 3, 4});
    auto tensor = core::Tensor::create(shape, core::DataType::FLOAT32);
    
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape().numel(), 24);
    
    core::Shape new_shape({6, 4});
    tensor->reshape(new_shape);
    
    EXPECT_EQ(tensor->shape().ndim(), 2);
    EXPECT_EQ(tensor->shape()[0], 6);
    EXPECT_EQ(tensor->shape()[1], 4);
    EXPECT_EQ(tensor->shape().numel(), 24);
}

TEST_F(TensorTest, TensorReshapeToVector) {
    core::Shape shape({2, 3, 4});
    auto tensor = core::Tensor::create(shape, core::DataType::FLOAT32);
    
    core::Shape new_shape({24});
    tensor->reshape(new_shape);
    
    EXPECT_EQ(tensor->shape().ndim(), 1);
    EXPECT_EQ(tensor->shape()[0], 24);
}

// ============================================================================
// Tensor Size Tests
// ============================================================================

TEST_F(TensorTest, TensorSizeInBytes) {
    core::Shape shape({10, 10});
    auto tensor = core::Tensor::create(shape, core::DataType::FLOAT32);
    
    ASSERT_NE(tensor, nullptr);
    size_t expected_bytes = 100 * sizeof(float);
    EXPECT_EQ(tensor->size_in_bytes(), expected_bytes);
}

TEST_F(TensorTest, TensorSizeInBytesInt32) {
    core::Shape shape({5, 4});
    auto tensor = core::Tensor::create(shape, core::DataType::INT32);
    
    ASSERT_NE(tensor, nullptr);
    size_t expected_bytes = 20 * sizeof(int32_t);
    EXPECT_EQ(tensor->size_in_bytes(), expected_bytes);
}

// ============================================================================
// Tensor Data Type Tests
// ============================================================================

TEST_F(TensorTest, DifferentDataTypes) {
    core::Shape shape({5, 5});
    
    auto float32_tensor = core::Tensor::create(shape, core::DataType::FLOAT32);
    EXPECT_EQ(float32_tensor->dtype(), core::DataType::FLOAT32);
    
    auto int32_tensor = core::Tensor::create(shape, core::DataType::INT32);
    EXPECT_EQ(int32_tensor->dtype(), core::DataType::INT32);
    
    auto int8_tensor = core::Tensor::create(shape, core::DataType::INT8);
    EXPECT_EQ(int8_tensor->dtype(), core::DataType::INT8);
}

