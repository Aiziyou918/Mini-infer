# Core module
set(CORE_SOURCES
    tensor.cpp
    allocator.cpp
    types.cpp
    op_type.cpp
)

add_library(mini_infer_core ${CORE_SOURCES})

target_include_directories(mini_infer_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mini_infer_core
    PUBLIC
        Threads::Threads
)

# Add CUDA support if enabled
if(MINI_INFER_ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_compile_definitions(mini_infer_core PUBLIC MINI_INFER_USE_CUDA)
    target_include_directories(mini_infer_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(mini_infer_core PUBLIC CUDA::cudart CUDA::cuda_driver)
endif()

# Set library properties
set_target_properties(mini_infer_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "mini_infer_core"
)

# Install
install(TARGETS mini_infer_core
    EXPORT Mini-InferTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

