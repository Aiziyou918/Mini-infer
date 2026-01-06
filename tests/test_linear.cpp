#include <gtest/gtest.h>

#include "mini_infer/core/tensor.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/operators/plugin_base.h"

using namespace mini_infer;

class LinearPluginTest : public ::testing::Test {
   protected:
    void SetUp() override {
        linear_plugin_ = operators::PluginRegistry::instance().create_plugin(
            core::OpType::kLINEAR, core::DeviceType::CPU);
    }
    void TearDown() override {
        linear_plugin_.reset();
    }

    std::unique_ptr<operators::IPlugin> linear_plugin_;
};

// ============================================================================
// Linear Plugin Tests
// ============================================================================

TEST_F(LinearPluginTest, BasicForward) {
    ASSERT_NE(linear_plugin_, nullptr);

    // Set parameters: in_features=3, out_features=2, use_bias=true
    auto param = std::make_shared<operators::LinearParam>(3, 2, true);
    linear_plugin_->set_param(param);

    core::Shape input_shape({2, 3});
    auto input = core::Tensor::create(input_shape, core::DataType::FLOAT32);
    ASSERT_NE(input, nullptr);

    float* input_data = static_cast<float*>(input->data());
    input_data[0] = 1.0f;
    input_data[1] = 2.0f;
    input_data[2] = 3.0f;
    input_data[3] = 4.0f;
    input_data[4] = 5.0f;
    input_data[5] = 6.0f;

    core::Shape weight_shape({2, 3});
    auto weight = core::Tensor::create(weight_shape, core::DataType::FLOAT32);
    ASSERT_NE(weight, nullptr);

    float* weight_data = static_cast<float*>(weight->data());
    weight_data[0] = 1.0f;
    weight_data[1] = 0.0f;
    weight_data[2] = -1.0f;
    weight_data[3] = 0.0f;
    weight_data[4] = 1.0f;
    weight_data[5] = 0.0f;

    core::Shape bias_shape({2});
    auto bias = core::Tensor::create(bias_shape, core::DataType::FLOAT32);
    ASSERT_NE(bias, nullptr);

    float* bias_data = static_cast<float*>(bias->data());
    bias_data[0] = 0.5f;
    bias_data[1] = -0.5f;

    // Infer shape
    std::vector<core::Shape> input_shapes = {input_shape, weight_shape, bias_shape};
    std::vector<core::Shape> output_shapes;
    auto status = linear_plugin_->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0][0], 2);  // batch
    EXPECT_EQ(output_shapes[0][1], 2);  // out_features

    auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

    std::vector<std::shared_ptr<core::Tensor>> inputs = {input, weight, bias};
    std::vector<std::shared_ptr<core::Tensor>> outputs = {output};

    operators::PluginContext ctx;
    status = linear_plugin_->enqueue(inputs, outputs, ctx);
    EXPECT_EQ(status, core::Status::SUCCESS);

    // Verify: output = input @ weight.T + bias
    // For first row: [1,2,3] @ [[1,0,-1],[0,1,0]].T + [0.5,-0.5]
    //              = [1*1+2*0+3*(-1), 1*0+2*1+3*0] + [0.5,-0.5]
    //              = [-2, 2] + [0.5, -0.5] = [-1.5, 1.5]
    float* output_data = static_cast<float*>(output->data());
    EXPECT_NEAR(output_data[0], -1.5f, 1e-5f);
    EXPECT_NEAR(output_data[1], 1.5f, 1e-5f);
}

TEST_F(LinearPluginTest, InferShape) {
    ASSERT_NE(linear_plugin_, nullptr);

    auto param = std::make_shared<operators::LinearParam>(64, 128, true);
    linear_plugin_->set_param(param);

    std::vector<core::Shape> input_shapes = {
        core::Shape({4, 64}),      // input
        core::Shape({128, 64}),    // weight
        core::Shape({128})         // bias
    };
    std::vector<core::Shape> output_shapes;

    auto status = linear_plugin_->infer_output_shapes(input_shapes, output_shapes);
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0].ndim(), 2);
    EXPECT_EQ(output_shapes[0][0], 4);
    EXPECT_EQ(output_shapes[0][1], 128);
}

TEST_F(LinearPluginTest, PluginClone) {
    ASSERT_NE(linear_plugin_, nullptr);

    auto cloned = linear_plugin_->clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->get_op_type(), core::OpType::kLINEAR);
    EXPECT_EQ(cloned->get_device_type(), core::DeviceType::CPU);
}
