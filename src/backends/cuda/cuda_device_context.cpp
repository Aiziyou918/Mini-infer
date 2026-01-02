#include "mini_infer/backends/cuda/cuda_device_context.h"

#ifdef MINI_INFER_USE_CUDA

#include "mini_infer/utils/logger.h"
#include <sstream>

namespace mini_infer {
namespace backends {
namespace cuda {

CUDADeviceContext::CUDADeviceContext(int device_id)
    : device_id_(device_id) {

    // Set device
    CUDA_CHECK_THROW(cudaSetDevice(device_id_));

    // Query device properties
    query_device_properties();

    // Initialize stream
    init_stream();

    MI_LOG_INFO("[CUDADeviceContext] Initialized on device " +
                std::to_string(device_id_) + ": " +
                std::string(device_prop_.name));
}

CUDADeviceContext::~CUDADeviceContext() {
    // Free workspace
    if (workspace_) {
        cudaFree(workspace_);
        workspace_ = nullptr;
    }

    // Destroy cuBLAS handle
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }

    // Destroy cuDNN handle
    if (cudnn_handle_) {
        cudnnDestroy(cudnn_handle_);
        cudnn_handle_ = nullptr;
    }

    // Destroy stream
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    MI_LOG_INFO("[CUDADeviceContext] Destroyed context for device " +
                std::to_string(device_id_));
}

void CUDADeviceContext::synchronize() {
    cudaError_t status = cudaStreamSynchronize(stream_);
    if (status != cudaSuccess) {
        MI_LOG_ERROR("[CUDADeviceContext] Stream synchronization failed: " +
           std::string(cudaGetErrorString(status)));
    }
}

cudnnHandle_t CUDADeviceContext::cudnn_handle() {
    if (!cudnn_handle_) {
        init_cudnn();
    }
    return cudnn_handle_;
}

cublasHandle_t CUDADeviceContext::cublas_handle() {
    if (!cublas_handle_) {
        init_cublas();
    }
    return cublas_handle_;
}

void* CUDADeviceContext::get_workspace(size_t size) {
    if (size > workspace_size_) {
        // Need to reallocate
        if (workspace_) {
            cudaFree(workspace_);
            workspace_ = nullptr;
        }

        cudaError_t status = cudaMalloc(&workspace_, size);
        if (status != cudaSuccess) {
            MI_LOG_ERROR("[CUDADeviceContext] Failed to allocate workspace of size " +
                         std::to_string(size) + " bytes: " +
                         std::string(cudaGetErrorString(status)));
            workspace_size_ = 0;
            return nullptr;
        }

        workspace_size_ = size;
        MI_LOG_INFO("[CUDADeviceContext] Allocated workspace: " +
                    std::to_string(size / 1024.0 / 1024.0) + " MB");
    }

    return workspace_;
}

size_t CUDADeviceContext::available_memory() const {
    size_t free_bytes = 0;
    size_t total_bytes = 0;

    cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        MI_LOG_WARNING("[CUDADeviceContext] Failed to query memory info");
        return 0;
    }

    return free_bytes;
}

void CUDADeviceContext::init_stream() {
    CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

void CUDADeviceContext::init_cudnn() {
    CUDNN_CHECK_THROW(cudnnCreate(&cudnn_handle_));
    CUDNN_CHECK_THROW(cudnnSetStream(cudnn_handle_, stream_));
    MI_LOG_INFO("[CUDADeviceContext] cuDNN handle initialized");
}

void CUDADeviceContext::init_cublas() {
    CUBLAS_CHECK_THROW(cublasCreate(&cublas_handle_));
    CUBLAS_CHECK_THROW(cublasSetStream(cublas_handle_, stream_));
    MI_LOG_INFO("[CUDADeviceContext] cuBLAS handle initialized");
}

void CUDADeviceContext::query_device_properties() {
    CUDA_CHECK_THROW(cudaGetDeviceProperties(&device_prop_, device_id_));

    // Log device info
    std::ostringstream oss;
    oss << "[CUDADeviceContext] Device " << device_id_ << " properties:\n"
        << "  Name: " << device_prop_.name << "\n"
        << "  Compute Capability: " << device_prop_.major << "." << device_prop_.minor << "\n"
        << "  Total Global Memory: " << (device_prop_.totalGlobalMem / 1024.0 / 1024.0) << " MB\n"
        << "  Multiprocessors: " << device_prop_.multiProcessorCount << "\n"
        << "  Max Threads Per Block: " << device_prop_.maxThreadsPerBlock << "\n"
        << "  Warp Size: " << device_prop_.warpSize;

    MI_LOG_INFO(oss.str());
}

}  // namespace cuda
}  // namespace backends
}  // namespace mini_infer

#endif  // MINI_INFER_USE_CUDA
