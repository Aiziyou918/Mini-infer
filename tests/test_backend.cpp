#include "mini_infer/backends/cpu_backend.h"
#include "mini_infer/backends/backend.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace mini_infer;

class BackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = backends::BackendFactory::create_backend(core::DeviceType::CPU);
    }

    void TearDown() override {
        backend_.reset();
    }

    std::shared_ptr<backends::Backend> backend_;
};

// ============================================================================
// CPU Backend Creation Tests
// ============================================================================

TEST_F(BackendTest, CPUBackendCreation) {
    ASSERT_NE(backend_, nullptr);
    EXPECT_EQ(backend_->device_type(), core::DeviceType::CPU);
}

TEST_F(BackendTest, BackendFactoryCreation) {
    auto cpu_backend = backends::BackendFactory::create_backend(core::DeviceType::CPU);
    ASSERT_NE(cpu_backend, nullptr);
    EXPECT_EQ(cpu_backend->device_type(), core::DeviceType::CPU);
}

// ============================================================================
// Memory Allocation Tests
// ============================================================================

TEST_F(BackendTest, BasicAllocation) {
    size_t size = 1024;
    void* ptr = backend_->allocate(size);
    
    ASSERT_NE(ptr, nullptr);
    
    backend_->deallocate(ptr);
}

TEST_F(BackendTest, LargeAllocation) {
    size_t size = 10 * 1024 * 1024; // 10 MB
    void* ptr = backend_->allocate(size);
    
    ASSERT_NE(ptr, nullptr);
    
    backend_->deallocate(ptr);
}

TEST_F(BackendTest, MultipleAllocations) {
    size_t size = 1024;
    void* ptr1 = backend_->allocate(size);
    void* ptr2 = backend_->allocate(size);
    void* ptr3 = backend_->allocate(size);
    
    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);
    ASSERT_NE(ptr3, nullptr);
    
    // Ensure different allocations
    EXPECT_NE(ptr1, ptr2);
    EXPECT_NE(ptr2, ptr3);
    EXPECT_NE(ptr1, ptr3);
    
    backend_->deallocate(ptr1);
    backend_->deallocate(ptr2);
    backend_->deallocate(ptr3);
}

// ============================================================================
// Memory Set Tests
// ============================================================================

TEST_F(BackendTest, MemsetZero) {
    size_t size = 1024;
    void* ptr = backend_->allocate(size);
    ASSERT_NE(ptr, nullptr);
    
    backend_->memset(ptr, 0, size);
    
    // Verify all bytes are zero
    uint8_t* data = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], 0);
    }
    
    backend_->deallocate(ptr);
}

TEST_F(BackendTest, MemsetValue) {
    size_t size = 128;
    void* ptr = backend_->allocate(size);
    ASSERT_NE(ptr, nullptr);
    
    uint8_t value = 0xAB;
    backend_->memset(ptr, value, size);
    
    // Verify all bytes are set to value
    uint8_t* data = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], value);
    }
    
    backend_->deallocate(ptr);
}

// ============================================================================
// Memory Copy Tests
// ============================================================================

TEST_F(BackendTest, MemcpyBasic) {
    size_t size = 100 * sizeof(float);
    void* src = backend_->allocate(size);
    void* dst = backend_->allocate(size);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    // Fill source data
    float* src_data = static_cast<float*>(src);
    for (int i = 0; i < 100; ++i) {
        src_data[i] = static_cast<float>(i);
    }
    
    // Copy
    backend_->memcpy(dst, src, size);
    
    // Verify
    float* dst_data = static_cast<float*>(dst);
    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(dst_data[i], static_cast<float>(i));
    }
    
    backend_->deallocate(src);
    backend_->deallocate(dst);
}

TEST_F(BackendTest, MemcpyInt32) {
    size_t count = 50;
    size_t size = count * sizeof(int32_t);
    void* src = backend_->allocate(size);
    void* dst = backend_->allocate(size);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    // Fill source data
    int32_t* src_data = static_cast<int32_t*>(src);
    for (size_t i = 0; i < count; ++i) {
        src_data[i] = static_cast<int32_t>(i * 2);
    }
    
    // Copy
    backend_->memcpy(dst, src, size);
    
    // Verify
    int32_t* dst_data = static_cast<int32_t*>(dst);
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(dst_data[i], static_cast<int32_t>(i * 2));
    }
    
    backend_->deallocate(src);
    backend_->deallocate(dst);
}

TEST_F(BackendTest, MemcpyLarge) {
    size_t size = 1024 * 1024; // 1 MB
    void* src = backend_->allocate(size);
    void* dst = backend_->allocate(size);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    // Fill with pattern
    uint8_t* src_data = static_cast<uint8_t*>(src);
    for (size_t i = 0; i < size; ++i) {
        src_data[i] = static_cast<uint8_t>(i % 256);
    }
    
    // Copy
    backend_->memcpy(dst, src, size);
    
    // Verify
    uint8_t* dst_data = static_cast<uint8_t*>(dst);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(dst_data[i], static_cast<uint8_t>(i % 256));
    }
    
    backend_->deallocate(src);
    backend_->deallocate(dst);
}

// ============================================================================
// Synchronization Tests
// ============================================================================

TEST_F(BackendTest, Synchronize) {
    // For CPU backend, synchronize should always succeed
    EXPECT_NO_THROW(backend_->synchronize());
}

