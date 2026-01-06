#include <gtest/gtest.h>

#include "mini_infer/core/tensor.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/operators/plugin_base.h"

using namespace mini_infer;

class PoolingPluginTest : public ::testing::Test {
   protected:
    void SetUp() override {
        pooling_plugin_ = operators::PluginRegistry::instance().create_plugin(
            core::OpType::kMAX_POOL, core::DeviceType::CPU);
    }
    void TearDown() override {
        pooling_plugin_.reset();
    }

    std::unique_ptr<operators::IPlugin> pooling_plugin_;
};

TEST_F(PoolingPluginTest, MaxPoolBasic) {
    ASSERT_NE(pooling_plugin_, nullptr);

    auto param = std::make_shared<operators::PoolingParam>(
        operators::PoolingType::MAX, 2, 2, 2, 2, 0, 0);
    pooling_plugin_->set_param(param);

    core::Shape shape({1, 1, 4, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    float* input_data = static_cast<float*>(input->data());
    for (int i = 0; i < 16; ++i) {
        input_data[i] = static_cast<float>(i);
    }

    // Infer output shape
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    auto status = pooling_plugin_->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0][0], 1);
    EXPECT_EQ(output_shapes[0][1], 1);
    EXPECT_EQ(output_shapes[0][2], 2);
    EXPECT_EQ(output_shapes[0][3], 2);

    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    operators::PluginContext ctx;
    status = pooling_plugin_->enqueue(inputs, outputs, ctx);
    EXPECT_EQ(status, core::Status::SUCCESS);

    float* output_data = static_cast<float*>(output->data());
    EXPECT_FLOAT_EQ(output_data[0], 5.0f);
    EXPECT_FLOAT_EQ(output_data[1], 7.0f);
    EXPECT_FLOAT_EQ(output_data[2], 13.0f);
    EXPECT_FLOAT_EQ(output_data[3], 15.0f);
}

TEST_F(PoolingPluginTest, AvgPoolBasic) {
    // Create AvgPool plugin
    auto avg_plugin = operators::PluginRegistry::instance().create_plugin(
        core::OpType::kAVERAGE_POOL, core::DeviceType::CPU);
    ASSERT_NE(avg_plugin, nullptr);

    auto param = std::make_shared<operators::PoolingParam>(
        operators::PoolingType::AVERAGE, 2, 2, 2, 2, 0, 0);
    avg_plugin->set_param(param);

    core::Shape shape({1, 1, 4, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    float* input_data = static_cast<float*>(input->data());
    for (int i = 0; i < 16; ++i) {
        input_data[i] = static_cast<float>(i);
    }

    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    auto status = avg_plugin->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);

    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    operators::PluginContext ctx;
    status = avg_plugin->enqueue(inputs, outputs, ctx);
    EXPECT_EQ(status, core::Status::SUCCESS);

    float* output_data = static_cast<float*>(output->data());
    // Top-left 2x2 average: (0+1+4+5)/4 = 2.5
    EXPECT_NEAR(output_data[0], 2.5f, 1e-5f);
    // Top-right 2x2 average: (2+3+6+7)/4 = 4.5
    EXPECT_NEAR(output_data[1], 4.5f, 1e-5f);
}

TEST_F(PoolingPluginTest, InferShapeWithPadding) {
    ASSERT_NE(pooling_plugin_, nullptr);

    auto param = std::make_shared<operators::PoolingParam>(
        operators::PoolingType::MAX, 3, 3, 2, 2, 1, 1);
    pooling_plugin_->set_param(param);

    std::vector<core::Shape> input_shapes = {core::Shape({1, 64, 112, 112})};
    std::vector<core::Shape> output_shapes;

    auto status = pooling_plugin_->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);

    EXPECT_EQ(output_shapes[0][0], 1);
    EXPECT_EQ(output_shapes[0][1], 64);
    EXPECT_EQ(output_shapes[0][2], 56);
    EXPECT_EQ(output_shapes[0][3], 56);
}

TEST_F(PoolingPluginTest, PluginClone) {
    ASSERT_NE(pooling_plugin_, nullptr);

    auto cloned = pooling_plugin_->clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->get_op_type(), core::OpType::kMAX_POOL);
    EXPECT_EQ(cloned->get_device_type(), core::DeviceType::CPU);
}
