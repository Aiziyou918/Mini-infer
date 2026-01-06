#include <gtest/gtest.h>

#include "mini_infer/core/tensor.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/operators/plugin_base.h"

using namespace mini_infer;

class FlattenPluginTest : public ::testing::Test {
   protected:
    void SetUp() override {
        flatten_plugin_ = operators::PluginRegistry::instance().create_plugin(
            core::OpType::kFLATTEN, core::DeviceType::CPU);
    }
    void TearDown() override {
        flatten_plugin_.reset();
    }

    std::unique_ptr<operators::IPlugin> flatten_plugin_;
};

TEST_F(FlattenPluginTest, BasicFlatten) {
    ASSERT_NE(flatten_plugin_, nullptr);

    auto param = std::make_shared<operators::FlattenParam>(1);
    flatten_plugin_->set_param(param);

    core::Shape shape({2, 3, 4, 5});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    // Infer output shape
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    auto status = flatten_plugin_->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0].ndim(), 2);
    EXPECT_EQ(output_shapes[0][0], 2);
    EXPECT_EQ(output_shapes[0][1], 60);

    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    operators::PluginContext ctx;
    status = flatten_plugin_->enqueue(inputs, outputs, ctx);
    EXPECT_EQ(status, core::Status::SUCCESS);
}

TEST_F(FlattenPluginTest, FlattenAxis0) {
    ASSERT_NE(flatten_plugin_, nullptr);

    auto param = std::make_shared<operators::FlattenParam>(0);
    flatten_plugin_->set_param(param);

    core::Shape shape({2, 3, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    auto status = flatten_plugin_->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    EXPECT_EQ(output_shapes[0].ndim(), 2);
    EXPECT_EQ(output_shapes[0][0], 1);
    EXPECT_EQ(output_shapes[0][1], 24);

    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    operators::PluginContext ctx;
    status = flatten_plugin_->enqueue(inputs, outputs, ctx);
    EXPECT_EQ(status, core::Status::SUCCESS);
}

TEST_F(FlattenPluginTest, FlattenLastAxis) {
    ASSERT_NE(flatten_plugin_, nullptr);

    auto param = std::make_shared<operators::FlattenParam>(2);
    flatten_plugin_->set_param(param);

    core::Shape shape({2, 3, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    auto status = flatten_plugin_->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    EXPECT_EQ(output_shapes[0].ndim(), 2);
    EXPECT_EQ(output_shapes[0][0], 6);
    EXPECT_EQ(output_shapes[0][1], 4);

    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    operators::PluginContext ctx;
    status = flatten_plugin_->enqueue(inputs, outputs, ctx);
    EXPECT_EQ(status, core::Status::SUCCESS);
}

TEST_F(FlattenPluginTest, PluginClone) {
    ASSERT_NE(flatten_plugin_, nullptr);

    auto cloned = flatten_plugin_->clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->get_op_type(), core::OpType::kFLATTEN);
    EXPECT_EQ(cloned->get_device_type(), core::DeviceType::CPU);
}
