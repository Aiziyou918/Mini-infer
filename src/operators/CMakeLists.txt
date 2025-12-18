# Operators module
set(OPERATORS_SOURCES
    operator.cpp
    conv2d.cpp
    relu.cpp
    linear.cpp
    pooling.cpp
    flatten.cpp
    reshape.cpp
)

add_library(mini_infer_operators ${OPERATORS_SOURCES})

target_include_directories(mini_infer_operators
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mini_infer_operators
    PUBLIC
        mini_infer_core
        mini_infer_kernels_all
        mini_infer_backends
)

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
