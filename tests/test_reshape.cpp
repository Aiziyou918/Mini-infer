#include <gtest/gtest.h>

#include "mini_infer/core/tensor.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/operators/plugin_base.h"

using namespace mini_infer;

class ReshapePluginTest : public ::testing::Test {
   protected:
    void SetUp() override {
        reshape_plugin_ = operators::PluginRegistry::instance().create_plugin(
            core::OpType::kRESHAPE, core::DeviceType::CPU);
    }
    void TearDown() override {
        reshape_plugin_.reset();
    }

    std::unique_ptr<operators::IPlugin> reshape_plugin_;
};

TEST_F(ReshapePluginTest, BasicReshape) {
    ASSERT_NE(reshape_plugin_, nullptr);

    auto param = std::make_shared<operators::ReshapeParam>(std::vector<int64_t>{2, 12});
    reshape_plugin_->set_param(param);

    core::Shape shape({2, 3, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    // Infer output shape
    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    auto status = reshape_plugin_->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0].ndim(), 2);
    EXPECT_EQ(output_shapes[0][0], 2);
    EXPECT_EQ(output_shapes[0][1], 12);

    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    operators::PluginContext ctx;
    status = reshape_plugin_->enqueue(inputs, outputs, ctx);
    EXPECT_EQ(status, core::Status::SUCCESS);
}

TEST_F(ReshapePluginTest, ReshapeWithInference) {
    ASSERT_NE(reshape_plugin_, nullptr);

    auto param = std::make_shared<operators::ReshapeParam>(std::vector<int64_t>{2, -1});
    reshape_plugin_->set_param(param);

    core::Shape shape({2, 3, 4});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    auto status = reshape_plugin_->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    EXPECT_EQ(output_shapes[0].ndim(), 2);
    EXPECT_EQ(output_shapes[0][0], 2);
    EXPECT_EQ(output_shapes[0][1], 12);

    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    operators::PluginContext ctx;
    status = reshape_plugin_->enqueue(inputs, outputs, ctx);
    EXPECT_EQ(status, core::Status::SUCCESS);
}

TEST_F(ReshapePluginTest, ReshapeToHigherDim) {
    ASSERT_NE(reshape_plugin_, nullptr);

    auto param = std::make_shared<operators::ReshapeParam>(std::vector<int64_t>{2, 3, 2, 2});
    reshape_plugin_->set_param(param);

    core::Shape shape({2, 12});
    auto input = core::Tensor::create(shape, core::DataType::FLOAT32);

    std::vector<core::Shape> input_shapes = {shape};
    std::vector<core::Shape> output_shapes;
    auto status = reshape_plugin_->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    EXPECT_EQ(output_shapes[0].ndim(), 4);
    EXPECT_EQ(output_shapes[0][0], 2);
    EXPECT_EQ(output_shapes[0][1], 3);
    EXPECT_EQ(output_shapes[0][2], 2);
    EXPECT_EQ(output_shapes[0][3], 2);

    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    operators::PluginContext ctx;
    status = reshape_plugin_->enqueue(inputs, outputs, ctx);
    EXPECT_EQ(status, core::Status::SUCCESS);
}

TEST_F(ReshapePluginTest, PluginClone) {
    ASSERT_NE(reshape_plugin_, nullptr);

    auto cloned = reshape_plugin_->clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->get_op_type(), core::OpType::kRESHAPE);
    EXPECT_EQ(cloned->get_device_type(), core::DeviceType::CPU);
}
