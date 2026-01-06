# Kernel implementations
# Note: Operator-level kernels (relu, conv2d, pooling, etc.) have been migrated to Plugin architecture.
# This library now only contains low-level utility kernels (GEMM, Im2Col, Bias, Transpose).

set(KERNEL_SOURCES
    cpu/gemm_cpu.cpp
    cpu/im2col_cpu.cpp
    cpu/bias_cpu.cpp
    cpu/transpose_cpu.cpp
)

# Add CUDA utility kernels if CUDA is enabled
if(MINI_INFER_ENABLE_CUDA AND CUDAToolkit_FOUND)
    list(APPEND KERNEL_SOURCES
        cuda/gemm_cuda.cu
        cuda/im2col_cuda.cu
        cuda/bias_cuda.cu
        cuda/transpose_cuda.cu
    )
endif()

add_library(mini_infer_kernels STATIC ${KERNEL_SOURCES})

target_include_directories(mini_infer_kernels
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mini_infer_kernels
    PUBLIC
        mini_infer_core
)

# Add CUDA support if enabled
if(MINI_INFER_ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_compile_definitions(mini_infer_kernels PUBLIC MINI_INFER_USE_CUDA)
    target_include_directories(mini_infer_kernels PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(mini_infer_kernels PUBLIC 
        CUDA::cudart 
        CUDA::cuda_driver
        CUDA::cublas
    )
    
    # Link cuDNN if available
    if(TARGET CUDA::cudnn)
        target_link_libraries(mini_infer_kernels PUBLIC CUDA::cudnn)
    endif()
    
    # Set CUDA properties for .cu files
    set_target_properties(mini_infer_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# Interface alias to keep kernel registry TU linked in static builds
add_library(mini_infer_kernels_all INTERFACE)
target_link_libraries(mini_infer_kernels_all
    INTERFACE
        $<$<OR:$<BOOL:${BUILD_SHARED_LIBS}>,$<BOOL:${MSVC}>>:mini_infer_kernels>
        $<$<AND:$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>,$<NOT:$<BOOL:${MSVC}>>>:-Wl,--whole-archive>
        $<$<AND:$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>,$<NOT:$<BOOL:${MSVC}>>>:$<TARGET_FILE:mini_infer_kernels>>
        $<$<AND:$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>,$<NOT:$<BOOL:${MSVC}>>>:-Wl,--no-whole-archive>
        $<$<AND:$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>,$<NOT:$<BOOL:${MSVC}>>>:mini_infer_kernels>
)
if (NOT BUILD_SHARED_LIBS AND MSVC)
    target_link_options(mini_infer_kernels_all INTERFACE "/WHOLEARCHIVE:mini_infer_kernels")
endif()

# Set library properties
set_target_properties(mini_infer_kernels PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "mini_infer_kernels"
)

# Install
install(TARGETS mini_infer_kernels
    EXPORT Mini-InferTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(TARGETS mini_infer_kernels_all
    EXPORT Mini-InferTargets
)
