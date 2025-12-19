# Kernel implementations

set(KERNEL_SOURCES
    cpu/gemm_cpu.cpp
    cpu/im2col_cpu.cpp
    cpu/bias_cpu.cpp
    cpu/conv2d_cpu.cpp
    cpu/transpose_cpu.cpp
    cpu/pooling_cpu.cpp
    cpu/relu_cpu.cpp
    cpu/flatten_cpu.cpp
    cpu/reshape_cpu.cpp
    cpu/softmax_cpu.cpp
)

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

# Interface alias to keep kernel registry TU linked in static builds
add_library(mini_infer_kernels_all INTERFACE)
target_link_libraries(mini_infer_kernels_all
    INTERFACE
        $<$<OR:$<BOOL:${BUILD_SHARED_LIBS}>,$<BOOL:${MSVC}>>:mini_infer_kernels>
        $<$<AND:$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>,$<NOT:$<BOOL:${MSVC}>>>:-Wl,--whole-archive>
        $<$<AND:$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>,$<NOT:$<BOOL:${MSVC}>>>:mini_infer_kernels>
        $<$<AND:$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>,$<NOT:$<BOOL:${MSVC}>>>:-Wl,--no-whole-archive>
)
if (NOT BUILD_SHARED_LIBS AND MSVC)
    target_link_options(mini_infer_kernels_all INTERFACE "/WHOLEARCHIVE:mini_infer_kernels")
endif()

# Future: Add CUDA kernels conditionally
# if(USE_CUDA)
#     enable_language(CUDA)
#     target_sources(mini_infer_kernels PRIVATE
#         cuda/gemm_cuda.cu
#         cuda/im2col_cuda.cu
#     )
# endif()

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
