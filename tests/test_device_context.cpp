#include <gtest/gtest.h>

#include "mini_infer/backends/device_context.h"


using namespace mini_infer;

class DeviceContextTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(DeviceContextTest, CreateCPUContext) {
    backends::DeviceContext ctx;
    ctx.device_type = core::DeviceType::CPU;
    ctx.device_id = 0;

    EXPECT_EQ(ctx.device_type, core::DeviceType::CPU);
    EXPECT_EQ(ctx.device_id, 0);
}

TEST_F(DeviceContextTest, CreateCUDAContext) {
    backends::DeviceContext ctx;
    ctx.device_type = core::DeviceType::CUDA;
    ctx.device_id = 0;

    EXPECT_EQ(ctx.device_type, core::DeviceType::CUDA);
    EXPECT_EQ(ctx.device_id, 0);
}

TEST_F(DeviceContextTest, MultipleDeviceIds) {
    for (int i = 0; i < 8; ++i) {
        backends::DeviceContext ctx;
        ctx.device_type = core::DeviceType::CUDA;
        ctx.device_id = i;

        EXPECT_EQ(ctx.device_id, i);
    }
}

TEST_F(DeviceContextTest, DefaultValues) {
    backends::DeviceContext ctx;

    EXPECT_EQ(ctx.device_type, core::DeviceType::CPU);
    EXPECT_EQ(ctx.device_id, 0);
}

TEST_F(DeviceContextTest, CopyContext) {
    backends::DeviceContext ctx1;
    ctx1.device_type = core::DeviceType::CUDA;
    ctx1.device_id = 2;

    backends::DeviceContext ctx2 = ctx1;

    EXPECT_EQ(ctx2.device_type, core::DeviceType::CUDA);
    EXPECT_EQ(ctx2.device_id, 2);
}

TEST_F(DeviceContextTest, AssignContext) {
    backends::DeviceContext ctx1;
    ctx1.device_type = core::DeviceType::CUDA;
    ctx1.device_id = 3;

    backends::DeviceContext ctx2;
    ctx2 = ctx1;

    EXPECT_EQ(ctx2.device_type, core::DeviceType::CUDA);
    EXPECT_EQ(ctx2.device_id, 3);
}

TEST_F(DeviceContextTest, MultipleContexts) {
    std::vector<backends::DeviceContext> contexts;

    for (int i = 0; i < 4; ++i) {
        backends::DeviceContext ctx;
        ctx.device_type = (i % 2 == 0) ? core::DeviceType::CPU : core::DeviceType::CUDA;
        ctx.device_id = i;
        contexts.push_back(ctx);
    }

    for (size_t i = 0; i < contexts.size(); ++i) {
        EXPECT_EQ(contexts[i].device_id, static_cast<int>(i));
    }
}
