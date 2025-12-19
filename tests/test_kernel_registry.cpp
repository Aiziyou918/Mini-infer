#include <gtest/gtest.h>

#include "mini_infer/core/op_type.h"
#include "mini_infer/kernels/kernel_registry.h"

using namespace mini_infer;

class KernelRegistryTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}
};

void dummy_kernel_func(kernels::KernelContext* ctx) {
    (void)ctx;  // Suppress unused parameter warning
}

void another_kernel_func(kernels::KernelContext* ctx) {
    (void)ctx;  // Suppress unused parameter warning
}

TEST_F(KernelRegistryTest, RegisterAndFind) {
    auto& registry = kernels::KernelRegistry::instance();

    registry.register_kernel(core::OpType::kRELU, core::DeviceType::CPU, core::DataType::FLOAT32,
                             dummy_kernel_func);

    auto func = registry.find(core::OpType::kRELU, core::DeviceType::CPU, core::DataType::FLOAT32);

    ASSERT_NE(func, nullptr);
}

TEST_F(KernelRegistryTest, FindNonExistent) {
    auto& registry = kernels::KernelRegistry::instance();

    auto func =
        registry.find(core::OpType::kUNKNOWN, core::DeviceType::CPU, core::DataType::FLOAT32);

    EXPECT_EQ(func, nullptr);
}

TEST_F(KernelRegistryTest, RegisterMultipleKernels) {
    auto& registry = kernels::KernelRegistry::instance();

    registry.register_kernel(core::OpType::kCONVOLUTION, core::DeviceType::CPU,
                             core::DataType::FLOAT32, dummy_kernel_func);

    registry.register_kernel(core::OpType::kCONVOLUTION, core::DeviceType::CPU,
                             core::DataType::INT32, another_kernel_func);

    auto func1 =
        registry.find(core::OpType::kCONVOLUTION, core::DeviceType::CPU, core::DataType::FLOAT32);

    auto func2 =
        registry.find(core::OpType::kCONVOLUTION, core::DeviceType::CPU, core::DataType::INT32);

    ASSERT_NE(func1, nullptr);
    ASSERT_NE(func2, nullptr);
}

TEST_F(KernelRegistryTest, DifferentDeviceTypes) {
    auto& registry = kernels::KernelRegistry::instance();

    registry.register_kernel(core::OpType::kGEMM, core::DeviceType::CPU, core::DataType::FLOAT32,
                             dummy_kernel_func);

    auto cpu_func =
        registry.find(core::OpType::kGEMM, core::DeviceType::CPU, core::DataType::FLOAT32);

    auto gpu_func =
        registry.find(core::OpType::kGEMM, core::DeviceType::CUDA, core::DataType::FLOAT32);

    ASSERT_NE(cpu_func, nullptr);
    EXPECT_EQ(gpu_func, nullptr);
}

TEST_F(KernelRegistryTest, OverwriteKernel) {
    auto& registry = kernels::KernelRegistry::instance();

    registry.register_kernel(core::OpType::kSOFTMAX, core::DeviceType::CPU, core::DataType::FLOAT32,
                             dummy_kernel_func);

    registry.register_kernel(core::OpType::kSOFTMAX, core::DeviceType::CPU, core::DataType::FLOAT32,
                             another_kernel_func);

    auto func =
        registry.find(core::OpType::kSOFTMAX, core::DeviceType::CPU, core::DataType::FLOAT32);

    ASSERT_NE(func, nullptr);
}

TEST_F(KernelRegistryTest, KernelContextParam) {
    kernels::KernelContext ctx;

    int test_param = 42;
    ctx.op_param = &test_param;

    const int* param = ctx.param<int>();
    ASSERT_NE(param, nullptr);
    EXPECT_EQ(*param, 42);
}

TEST_F(KernelRegistryTest, DeviceContextGetSet) {
    // Test device context get/set
    // Note: DeviceContext is abstract, so we just test the get/set functions
    kernels::set_current_device_context(nullptr);

    auto* retrieved_ctx = kernels::get_current_device_context();
    EXPECT_EQ(retrieved_ctx, nullptr);
}

TEST_F(KernelRegistryTest, MultipleDataTypes) {
    auto& registry = kernels::KernelRegistry::instance();

    std::vector<core::DataType> dtypes = {core::DataType::FLOAT32, core::DataType::INT32,
                                          core::DataType::INT8};

    for (auto dtype : dtypes) {
        registry.register_kernel(core::OpType::kPOOLING, core::DeviceType::CPU, dtype,
                                 dummy_kernel_func);
    }

    for (auto dtype : dtypes) {
        auto func = registry.find(core::OpType::kPOOLING, core::DeviceType::CPU, dtype);
        ASSERT_NE(func, nullptr);
    }
}

TEST_F(KernelRegistryTest, SingletonInstance) {
    auto& registry1 = kernels::KernelRegistry::instance();
    auto& registry2 = kernels::KernelRegistry::instance();

    EXPECT_EQ(&registry1, &registry2);
}
