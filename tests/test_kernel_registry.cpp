#include <gtest/gtest.h>

#include "mini_infer/core/op_type.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/operators/cpu_plugin.h"

using namespace mini_infer;

class PluginRegistryTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Simple test plugin for testing
class TestCPUPlugin : public operators::SimpleCPUPlugin<TestCPUPlugin> {
public:
    TestCPUPlugin() = default;
    ~TestCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "TestOp";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kRELU;
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {
        output_shapes = input_shapes;
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const operators::PluginContext& context) override {
        (void)inputs;
        (void)outputs;
        (void)context;
        return core::Status::SUCCESS;
    }
};

TEST_F(PluginRegistryTest, CreatePlugin) {
    auto& registry = operators::PluginRegistry::instance();

    // ReLU CPU plugin should be registered by the static initializer
    auto plugin = registry.create_plugin(core::OpType::kRELU, core::DeviceType::CPU);

    ASSERT_NE(plugin, nullptr);
    EXPECT_EQ(plugin->get_op_type(), core::OpType::kRELU);
    EXPECT_EQ(plugin->get_device_type(), core::DeviceType::CPU);
}

TEST_F(PluginRegistryTest, CreateNonExistentPlugin) {
    auto& registry = operators::PluginRegistry::instance();

    auto plugin = registry.create_plugin(core::OpType::kUNKNOWN, core::DeviceType::CPU);

    EXPECT_EQ(plugin, nullptr);
}

TEST_F(PluginRegistryTest, HasPlugin) {
    auto& registry = operators::PluginRegistry::instance();

    // ReLU CPU should be registered
    EXPECT_TRUE(registry.has_plugin(core::OpType::kRELU, core::DeviceType::CPU));

    // Unknown should not be registered
    EXPECT_FALSE(registry.has_plugin(core::OpType::kUNKNOWN, core::DeviceType::CPU));
}

TEST_F(PluginRegistryTest, CreatePluginByTypeName) {
    auto& registry = operators::PluginRegistry::instance();

    auto plugin = registry.create_plugin("Relu", core::DeviceType::CPU);

    ASSERT_NE(plugin, nullptr);
    EXPECT_EQ(plugin->get_op_type(), core::OpType::kRELU);
}

TEST_F(PluginRegistryTest, GetRegisteredKeys) {
    auto& registry = operators::PluginRegistry::instance();

    auto keys = registry.get_registered_keys();

    // Should have at least the ReLU plugin registered
    EXPECT_GT(keys.size(), 0);

    // Check that ReLU CPU is in the list
    bool found_relu_cpu = false;
    for (const auto& key : keys) {
        if (key.op_type == core::OpType::kRELU && key.device_type == core::DeviceType::CPU) {
            found_relu_cpu = true;
            break;
        }
    }
    EXPECT_TRUE(found_relu_cpu);
}

TEST_F(PluginRegistryTest, PluginClone) {
    auto& registry = operators::PluginRegistry::instance();

    auto plugin = registry.create_plugin(core::OpType::kRELU, core::DeviceType::CPU);
    ASSERT_NE(plugin, nullptr);

    auto cloned = plugin->clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->get_op_type(), plugin->get_op_type());
    EXPECT_EQ(cloned->get_device_type(), plugin->get_device_type());
}

TEST_F(PluginRegistryTest, SingletonInstance) {
    auto& registry1 = operators::PluginRegistry::instance();
    auto& registry2 = operators::PluginRegistry::instance();

    EXPECT_EQ(&registry1, &registry2);
}

TEST_F(PluginRegistryTest, PluginShapeInference) {
    auto& registry = operators::PluginRegistry::instance();

    auto plugin = registry.create_plugin(core::OpType::kRELU, core::DeviceType::CPU);
    ASSERT_NE(plugin, nullptr);

    std::vector<core::Shape> input_shapes = {core::Shape({1, 3, 224, 224})};
    std::vector<core::Shape> output_shapes;

    auto status = plugin->infer_output_shapes(input_shapes, output_shapes);

    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], input_shapes[0]);
}
