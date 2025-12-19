#include <gtest/gtest.h>

#include "mini_infer/core/allocator.h"


using namespace mini_infer;

class AllocatorTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(AllocatorTest, BasicAllocation) {
    core::CPUAllocator allocator;

    size_t size = 1024;
    void* ptr = allocator.allocate(size);

    ASSERT_NE(ptr, nullptr);

    allocator.deallocate(ptr);
}

TEST_F(AllocatorTest, ZeroSizeAllocation) {
    core::CPUAllocator allocator;

    void* ptr = allocator.allocate(0);

    EXPECT_EQ(ptr, nullptr);
}

TEST_F(AllocatorTest, LargeAllocation) {
    core::CPUAllocator allocator;

    size_t size = 1024 * 1024 * 10;
    void* ptr = allocator.allocate(size);

    ASSERT_NE(ptr, nullptr);

    allocator.deallocate(ptr);
}

TEST_F(AllocatorTest, MultipleAllocations) {
    core::CPUAllocator allocator;

    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        void* ptr = allocator.allocate(1024);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    for (auto ptr : ptrs) {
        allocator.deallocate(ptr);
    }
}

TEST_F(AllocatorTest, WriteAndRead) {
    core::CPUAllocator allocator;

    size_t size = 100 * sizeof(float);
    void* ptr = allocator.allocate(size);
    ASSERT_NE(ptr, nullptr);

    float* data = static_cast<float*>(ptr);
    for (int i = 0; i < 100; ++i) {
        data[i] = static_cast<float>(i);
    }

    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
    }

    allocator.deallocate(ptr);
}

TEST_F(AllocatorTest, DeallocateNull) {
    core::CPUAllocator allocator;

    allocator.deallocate(nullptr);
}

TEST_F(AllocatorTest, AlignedAllocation) {
    core::CPUAllocator allocator;

    size_t size = 1024;
    void* ptr = allocator.allocate(size);

    ASSERT_NE(ptr, nullptr);

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % alignof(std::max_align_t), 0);

    allocator.deallocate(ptr);
}

TEST_F(AllocatorTest, DifferentSizes) {
    core::CPUAllocator allocator;

    std::vector<size_t> sizes = {1, 16, 64, 256, 1024, 4096, 65536};

    for (auto size : sizes) {
        void* ptr = allocator.allocate(size);
        ASSERT_NE(ptr, nullptr);
        allocator.deallocate(ptr);
    }
}

TEST_F(AllocatorTest, ReallocatePattern) {
    core::CPUAllocator allocator;

    for (int i = 0; i < 100; ++i) {
        void* ptr = allocator.allocate(1024);
        ASSERT_NE(ptr, nullptr);
        allocator.deallocate(ptr);
    }
}

TEST_F(AllocatorTest, DeviceType) {
    core::CPUAllocator allocator;

    EXPECT_EQ(allocator.device_type(), core::DeviceType::CPU);
}

TEST_F(AllocatorTest, ConcurrentAllocations) {
    core::CPUAllocator allocator;

    std::vector<void*> ptrs;
    for (int i = 0; i < 50; ++i) {
        void* ptr = allocator.allocate((i + 1) * 100);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    for (auto ptr : ptrs) {
        allocator.deallocate(ptr);
    }
}

TEST_F(AllocatorTest, MemoryPattern) {
    core::CPUAllocator allocator;

    size_t size = 256 * sizeof(int);
    void* ptr = allocator.allocate(size);
    ASSERT_NE(ptr, nullptr);

    int* data = static_cast<int*>(ptr);
    for (int i = 0; i < 256; ++i) {
        data[i] = i * 2;
    }

    for (int i = 0; i < 256; ++i) {
        EXPECT_EQ(data[i], i * 2);
    }

    allocator.deallocate(ptr);
}
