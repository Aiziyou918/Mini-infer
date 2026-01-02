#pragma once

#include "mini_infer/backends/device_context.h"
#include "mini_infer/core/types.h"

#ifdef MINI_INFER_USE_CUDA

#include "mini_infer/utils/logger.h"

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <string>

namespace mini_infer {
namespace backends {
namespace cuda {

/**
 * @brief CUDA error checking macro (returns Status)
 *
 * Checks CUDA API return status and logs error if failed.
 * Use in functions that return core::Status.
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            MI_LOG_ERROR("[CUDA] " + std::string(cudaGetErrorString(status)) + \
                         " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
            return core::Status::ERROR_RUNTIME; \
        } \
    } while(0)

/**
 * @brief CUDA error checking macro (throws exception)
 *
 * Checks CUDA API return status and throws exception if failed.
 * Use in constructors or functions that cannot return Status.
 */
#define CUDA_CHECK_THROW(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::string error_msg = "[CUDA] " + std::string(cudaGetErrorString(status)) + \
                                   " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__); \
            MI_LOG_ERROR(error_msg); \
            throw std::runtime_error(error_msg); \
        } \
    } while(0)

/**
 * @brief cuDNN error checking macro (returns Status)
 */
#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            MI_LOG_ERROR("[cuDNN] " + std::string(cudnnGetErrorString(status)) + \
                         " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
            return core::Status::ERROR_RUNTIME; \
        } \
    } while(0)

/**
 * @brief cuDNN error checking macro (throws exception)
 */
#define CUDNN_CHECK_THROW(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            std::string error_msg = "[cuDNN] " + std::string(cudnnGetErrorString(status)) + \
                                   " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__); \
            MI_LOG_ERROR(error_msg); \
            throw std::runtime_error(error_msg); \
        } \
    } while(0)

/**
 * @brief cuBLAS error checking macro (returns Status)
 */
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            MI_LOG_ERROR("[cuBLAS] Error code " + std::to_string(status) + \
                         " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
            return core::Status::ERROR_RUNTIME; \
        } \
    } while(0)

/**
 * @brief cuBLAS error checking macro (throws exception)
 */
#define CUBLAS_CHECK_THROW(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::string error_msg = "[cuBLAS] Error code " + std::to_string(status) + \
                                   " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__); \
            MI_LOG_ERROR(error_msg); \
            throw std::runtime_error(error_msg); \
        } \
    } while(0)

/**
 * @brief CUDA Device Context (TensorRT-style)
 *
 * Manages CUDA device resources including:
 * - CUDA stream for async execution
 * - cuDNN handle for deep learning operations
 * - cuBLAS handle for linear algebra operations
 * - Workspace memory for temporary buffers
 *
 * Design follows TensorRT's IExecutionContext pattern.
 */
class CUDADeviceContext : public DeviceContext {
public:
    /**
     * @brief Construct CUDA device context
     *
     * @param device_id CUDA device ID (default: 0)
     */
    explicit CUDADeviceContext(int device_id = 0);

    /**
     * @brief Destructor - cleans up all CUDA resources
     */
    ~CUDADeviceContext() override;

    // Disable copy
    CUDADeviceContext(const CUDADeviceContext&) = delete;
    CUDADeviceContext& operator=(const CUDADeviceContext&) = delete;

    // =========================================================================
    // DeviceContext Interface Implementation
    // =========================================================================

    /**
     * @brief Get device type
     * @return DeviceType::CUDA
     */
    core::DeviceType device_type() const override {
        return core::DeviceType::CUDA;
    }

    /**
     * @brief Synchronize device (wait for all operations to complete)
     */
    void synchronize() override;

    // =========================================================================
    // CUDA-Specific Interface
    // =========================================================================

    /**
     * @brief Get CUDA device ID
     * @return Device ID
     */
    int device_id() const { return device_id_; }

    /**
     * @brief Get CUDA stream
     * @return CUDA stream handle
     */
    cudaStream_t stream() const { return stream_; }

    /**
     * @brief Get cuDNN handle (lazy initialization)
     * @return cuDNN handle
     */
    cudnnHandle_t cudnn_handle();

    /**
     * @brief Get cuBLAS handle (lazy initialization)
     * @return cuBLAS handle
     */
    cublasHandle_t cublas_handle();

    /**
     * @brief Get workspace memory (TensorRT-style)
     *
     * Allocates or reuses workspace memory for temporary buffers.
     * If requested size is larger than current workspace, reallocates.
     *
     * @param size Required workspace size in bytes
     * @return Pointer to workspace memory
     */
    void* get_workspace(size_t size);

    /**
     * @brief Get currerkspace size
     * @return Workspace size in bytes
     */
    size_t workspace_size() const { return workspace_size_; }

    /**
     * @brief Query device properties
     * @return CUDA device properties
     */
    const cudaDeviceProp& device_properties() const { return device_prop_; }

    /**
     * @brief Get available memory on device
     * @return Available memory in bytes
     */
    size_t available_memory() const;

private:
    int device_id_;                      ///< CUDA device ID
    cudaStream_t stream_{nullptr};       ///< CUDA stream for async execution
    cudnnHandle_t cudnn_handle_{nullptr}; ///< cuDNN handle (lazy init)
    cublasHandle_t cublas_handle_{nullptr}; ///< cuBLAS handle (lazy init)
    cudaDeviceProp device_prop_;         ///< Device properties

    void* workspace_{nullptr};           ///< Workspace memory
    size_t workspace_size_{0};           ///< Current workspace size

    /**
     * @brief Initialize CUDA stream
     */
    void init_stream();

    /**
     * @brief Initialize cuDNN handle (lazy)
     */
    void init_cudnn();

    /**
     * @brief Initialize cuBLAS handle (lazy)
     */
    void init_cublas();

    /**
     * @brief Query and cache device properties
     */
    void query_device_properties();
};

}  // namespace cuda
}  // namespace backends
}  // namespace mini_infer

#endif  // MINI_INFER_USE_CUDA
