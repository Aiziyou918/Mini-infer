# Operators module - Plugin-based architecture
#
# All operators are now implemented as Plugins.
# The Operator base class is kept for backward compatibility with Node/Graph.

# Core infrastructure
set(OPERATORS_SOURCES
    operator.cpp
    plugin_registry.cpp
)

# Plugin implementations (CPU)
set(PLUGIN_CPU_SOURCES
    cpu/relu_cpu.cpp
    cpu/flatten_cpu.cpp
    cpu/reshape_cpu.cpp
    cpu/softmax_cpu.cpp
    cpu/pooling_cpu.cpp
    cpu/linear_cpu.cpp
    cpu/conv2d_cpu.cpp
    # BERT operators
    cpu/add_cpu.cpp
    cpu/mul_cpu.cpp
    cpu/div_cpu.cpp
    cpu/sqrt_cpu.cpp
    cpu/gelu_cpu.cpp
    cpu/transpose_cpu.cpp
    cpu/squeeze_cpu.cpp
    cpu/unsqueeze_cpu.cpp
    cpu/layernorm_cpu.cpp
    cpu/gather_cpu.cpp
    cpu/matmul_cpu.cpp
    cpu/shape_cpu.cpp
    cpu/concat_cpu.cpp
    # Additional BERT operators (LayerNorm decomposition)
    cpu/sub_cpu.cpp
    cpu/pow_cpu.cpp
    cpu/tanh_cpu.cpp
    cpu/erf_cpu.cpp
    cpu/cast_cpu.cpp
    cpu/reduce_mean_cpu.cpp
    cpu/slice_cpu.cpp
    # Comparison and logical operators
    cpu/constant_of_shape_cpu.cpp
    cpu/equal_cpu.cpp
    cpu/where_cpu.cpp
    cpu/expand_cpu.cpp
)

# Plugin implementations (CUDA)
set(PLUGIN_CUDA_SOURCES
    cuda/relu_cuda.cu
    cuda/flatten_cuda.cu
    cuda/reshape_cuda.cu
    cuda/softmax_cuda.cu
    cuda/pooling_cuda.cu
    cuda/linear_cuda.cu
    cuda/conv2d_cuda.cu
    # BERT operators
    cuda/add_cuda.cu
    cuda/mul_cuda.cu
    cuda/div_cuda.cu
    cuda/sqrt_cuda.cu
    cuda/gelu_cuda.cu
    cuda/transpose_cuda.cu
    cuda/squeeze_cuda.cu
    cuda/unsqueeze_cuda.cu
    cuda/layernorm_cuda.cu
    cuda/gather_cuda.cu
    cuda/matmul_cuda.cu
    # Additional BERT operators (LayerNorm decomposition)
    cuda/sub_cuda.cu
    cuda/pow_cuda.cu
    cuda/tanh_cuda.cu
    cuda/erf_cuda.cu
    cuda/cast_cuda.cu
    cuda/reduce_mean_cuda.cu
    cuda/slice_cuda.cu
)

# Combine sources
list(APPEND OPERATORS_SOURCES ${PLUGIN_CPU_SOURCES})

# Add CUDA plugin sources if CUDA is enabled
if(MINI_INFER_ENABLE_CUDA AND CUDAToolkit_FOUND)
    list(APPEND OPERATORS_SOURCES ${PLUGIN_CUDA_SOURCES})
endif()

add_library(mini_infer_operators ${OPERATORS_SOURCES})

target_include_directories(mini_infer_operators
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mini_infer_operators
    PUBLIC
        mini_infer_core
        mini_infer_kernels
        mini_infer_backends
)

# Add CUDA support for plugin .cu files
if(MINI_INFER_ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_compile_definitions(mini_infer_operators PUBLIC MINI_INFER_USE_CUDA)
    target_include_directories(mini_infer_operators PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(mini_infer_operators PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
    )

    set_target_properties(mini_infer_operators PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# Set library properties
set_target_properties(mini_infer_operators PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "mini_infer_operators"
)

# Install
install(TARGETS mini_infer_operators
    EXPORT Mini-InferTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
