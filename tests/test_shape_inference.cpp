#include <gtest/gtest.h>
#include "mini_infer/core/tensor.h"
#include "mini_infer/operators/conv2d.h"
#include "mini_infer/operators/linear.h"
#include "mini_infer/operators/relu.h"
#include "mini_infer/operators/pooling.h"
#include "mini_infer/operators/flatten.h"

using namespace mini_infer;

// ============================================================================
// Shape Utility Tests
// ============================================================================

TEST(ShapeTest, IsDynamic) {
    // Static shape
    core::Shape static_shape({1, 3, 224, 224});
    EXPECT_FALSE(static_shape.is_dynamic());
    
    // Dynamic batch size
    core::Shape dynamic_batch({-1, 3, 224, 224});
    EXPECT_TRUE(dynamic_batch.is_dynamic());
    
    // Dynamic in middle
    core::Shape dynamic_mid({1, -1, 224, 224});
    EXPECT_TRUE(dynamic_mid.is_dynamic());
}

// ============================================================================
// Conv2D Shape Inference Tests
// ============================================================================

TEST(Conv2DShapeInferenceTest, BasicConv) {
    operators::Conv2DParam param;
    param.stride_h = 1;
    param.stride_w = 1;
    param.padding_h = 0;
    param.padding_w = 0;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.groups = 1;
    param.use_bias = false;
    
    operators::Conv2D conv(param);
    
    // Input: [1, 3, 28, 28], Weight: [32, 3, 5, 5]
    std::vector<core::Shape> input_shapes = {
        core::Shape({1, 3, 28, 28}),  // data
        core::Shape({32, 3, 5, 5})    // weight
    };
    
    std::vector<core::Shape> output_shapes;
    auto status = conv.infer_shape(input_shapes, output_shapes);
    
    ASSERT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);
    
    // Output: [1, 32, 24, 24]
    EXPECT_EQ(output_shapes[0].ndim(), 4);
    EXPECT_EQ(output_shapes[0][0], 1);
    EXPECT_EQ(output_shapes[0][1], 32);
    EXPECT_EQ(output_shapes[0][2], 24);
    EXPECT_EQ(output_shapes[0][3], 24);
}

TEST(Conv2DShapeInferenceTest, ConvWithPadding) {
    operators::Conv2DParam param;
    param.stride_h = 1;
    param.stride_w = 1;
    param.padding_h = 2;
    param.padding_w = 2;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.groups = 1;
    
    operators::Conv2D conv(param);
    
    // Input: [1, 3, 28, 28], Weight: [64, 3, 5, 5]
    std::vector<core::Shape> input_shapes = {
        core::Shape({1, 3, 28, 28}),
        core::Shape({64, 3, 5, 5})
    };
    
    std::vector<core::Shape> output_shapes;
    auto status = conv.infer_shape(input_shapes, output_shapes);
    
    ASSERT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);
    
    // With padding=2, output size should be same as input
    // Output: [1, 64, 28, 28]
    EXPECT_EQ(output_shapes[0][2], 28);
    EXPECT_EQ(output_shapes[0][3], 28);
}

TEST(Conv2DShapeInferenceTest, ConvWithStride) {
    operators::Conv2DParam param;
    param.stride_h = 2;
    param.stride_w = 2;
    param.padding_h = 0;
    param.padding_w = 0;
    param.dilation_h = 1;
    param.dilation_w = 1;
    param.groups = 1;
    
    operators::Conv2D conv(param);
    
    // Input: [1, 64, 28, 28], Weight: [128, 64, 3, 3]
    std::vector<core::Shape> input_shapes = {
        core::Shape({1, 64, 28, 28}),
        core::Shape({128, 64, 3, 3})
    };
    
    std::vector<core::Shape> output_shapes;
    auto status = conv.infer_shape(input_shapes, output_shapes);
    
    ASSERT_EQ(status, core::Status::SUCCESS);
    
    // With stride=2, output size should be half
    // Output: [1, 128, 13, 13]
    EXPECT_EQ(output_shapes[0][2], 13);
    EXPECT_EQ(output_shapes[0][3], 13);
}

// ============================================================================
// Pooling Shape Inference Tests
// ============================================================================

TEST(PoolingShapeInferenceTest, MaxPool) {
    operators::PoolingParam param;
    param.type = operators::PoolingType::MAX;
    param.kernel_h = 2;
    param.kernel_w = 2;
    param.stride_h = 2;
    param.stride_w = 2;
    param.padding_h = 0;
    param.padding_w = 0;
    
    operators::Pooling pool(param);
    
    // Input: [1, 64, 28, 28]
    std::vector<core::Shape> input_shapes = {
        core::Shape({1, 64, 28, 28})
    };
    
    std::vector<core::Shape> output_shapes;
    auto status = pool.infer_shape(input_shapes, output_shapes);
    
    ASSERT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);
    
    // Output: [1, 64, 14, 14]
    EXPECT_EQ(output_shapes[0][0], 1);
    EXPECT_EQ(output_shapes[0][1], 64);
    EXPECT_EQ(output_shapes[0][2], 14);
    EXPECT_EQ(output_shapes[0][3], 14);
}

// ============================================================================
// Linear Shape Inference Tests
// ============================================================================

TEST(LinearShapeInferenceTest, Basic2D) {
    operators::LinearParam param;
    param.in_features = 512;
    param.out_features = 10;
    param.use_bias = true;
    
    operators::Linear linear(param);
    
    // Input: [32, 512], Weight: [10, 512], Bias: [10]
    std::vector<core::Shape> input_shapes = {
        core::Shape({32, 512}),
        core::Shape({10, 512}),
        core::Shape({10})
    };
    
    std::vector<core::Shape> output_shapes;
    auto status = linear.infer_shape(input_shapes, output_shapes);
    
    ASSERT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);
    
    // Output: [32, 10]
    EXPECT_EQ(output_shapes[0].ndim(), 2);
    EXPECT_EQ(output_shapes[0][0], 32);
    EXPECT_EQ(output_shapes[0][1], 10);
}

TEST(LinearShapeInferenceTest, MultiDimensional) {
    operators::LinearParam param;
    param.in_features = 128;
    param.out_features = 64;
    param.use_bias = false;
    
    operators::Linear linear(param);
    
    // Input: [4, 8, 128], Weight: [64, 128]
    std::vector<core::Shape> input_shapes = {
        core::Shape({4, 8, 128}),
        core::Shape({64, 128})
    };
    
    std::vector<core::Shape> output_shapes;
    auto status = linear.infer_shape(input_shapes, output_shapes);
    
    ASSERT_EQ(status, core::Status::SUCCESS);
    
    // Output: [4, 8, 64]
    EXPECT_EQ(output_shapes[0].ndim(), 3);
    EXPECT_EQ(output_shapes[0][0], 4);
    EXPECT_EQ(output_shapes[0][1], 8);
    EXPECT_EQ(output_shapes[0][2], 64);
}

// ============================================================================
// ReLU Shape Inference Tests
// ============================================================================

TEST(ReLUShapeInferenceTest, PreservesShape) {
    operators::ReLU relu;
    
    // Test various shapes
    std::vector<std::vector<int64_t>> test_shapes = {
        {1, 3, 224, 224},
        {32, 512},
        {1, 10},
        {4, 8, 16, 32}
    };
    
    for (const auto& dims : test_shapes) {
        std::vector<core::Shape> input_shapes = {core::Shape(dims)};
        std::vector<core::Shape> output_shapes;
        
        auto status = relu.infer_shape(input_shapes, output_shapes);
        
        ASSERT_EQ(status, core::Status::SUCCESS);
        ASSERT_EQ(output_shapes.size(), 1);
        EXPECT_EQ(output_shapes[0].dims(), dims);
    }
}

// ============================================================================
// Flatten Shape Inference Tests
// ============================================================================

TEST(FlattenShapeInferenceTest, DefaultAxis) {
    operators::FlattenParam param;
    param.axis = 1;  // Default: flatten from axis 1
    
    operators::Flatten flatten(param);
    
    // Input: [1, 64, 7, 7]
    std::vector<core::Shape> input_shapes = {
        core::Shape({1, 64, 7, 7})
    };
    
    std::vector<core::Shape> output_shapes;
    auto status = flatten.infer_shape(input_shapes, output_shapes);
    
    ASSERT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);
    
    // Output: [1, 3136] (64 * 7 * 7 = 3136)
    EXPECT_EQ(output_shapes[0].ndim(), 2);
    EXPECT_EQ(output_shapes[0][0], 1);
    EXPECT_EQ(output_shapes[0][1], 3136);
}

TEST(FlattenShapeInferenceTest, CustomAxis) {
    operators::FlattenParam param;
    param.axis = 2;
    
    operators::Flatten flatten(param);
    
    // Input: [2, 3, 4, 5]
    std::vector<core::Shape> input_shapes = {
        core::Shape({2, 3, 4, 5})
    };
    
    std::vector<core::Shape> output_shapes;
    auto status = flatten.infer_shape(input_shapes, output_shapes);
    
    ASSERT_EQ(status, core::Status::SUCCESS);
    
    // Output: [6, 20] (2*3=6, 4*5=20)
    EXPECT_EQ(output_shapes[0].ndim(), 2);
    EXPECT_EQ(output_shapes[0][0], 6);
    EXPECT_EQ(output_shapes[0][1], 20);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(ShapeInferenceErrorTest, Conv2DInvalidInputCount) {
    operators::Conv2DParam param;
    operators::Conv2D conv(param);
    
    // Only 1 input (need at least 2: data + weight)
    std::vector<core::Shape> input_shapes = {
        core::Shape({1, 3, 28, 28})
    };
    
    std::vector<core::Shape> output_shapes;
    auto status = conv.infer_shape(input_shapes, output_shapes);
    
    EXPECT_NE(status, core::Status::SUCCESS);
}

TEST(ShapeInferenceErrorTest, LinearShapeMismatch) {
    operators::LinearParam param;
    param.in_features = 512;
    param.out_features = 10;
    
    operators::Linear linear(param);
    
    // Weight shape doesn't match param
    std::vector<core::Shape> input_shapes = {
        core::Shape({32, 512}),
        core::Shape({10, 256})  // Wrong: should be [10, 512]
    };
    
    std::vector<core::Shape> output_shapes;
    auto status = linear.infer_shape(input_shapes, output_shapes);
    
    EXPECT_NE(status, core::Status::SUCCESS);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

