# Backends module
set(BACKENDS_SOURCES
    device_context.cpp
    cpu/cpu_allocator.cpp
)

# Add CUDA sources if enabled
if(MINI_INFER_ENABLE_CUDA AND CUDAToolkit_FOUND)
    list(APPEND BACKENDS_SOURCES
        cuda/cuda_allocator.cpp
        cuda/cuda_device_context.cpp
    )
endif()

add_library(mini_infer_backends ${BACKENDS_SOURCES})

target_include_directories(mini_infer_backends
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mini_infer_backends
    PUBLIC
        mini_infer_core
)

# Add CUDA support if enabled
if(MINI_INFER_ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_compile_definitions(mini_infer_backends PUBLIC MINI_INFER_USE_CUDA)
    target_include_directories(mini_infer_backends PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(mini_infer_backends PUBLIC CUDA::cudart CUDA::cuda_driver)
    
    if(CUDNN_LIBRARY)
        target_link_libraries(mini_infer_backends PUBLIC ${CUDNN_LIBRARY})
    endif()
endif()

# Set library properties
set_target_properties(mini_infer_backends PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "mini_infer_backends"
)

# Install
install(TARGETS mini_infer_backends
    EXPORT Mini-InferTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

