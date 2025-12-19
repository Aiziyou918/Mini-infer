# Backends module
set(BACKENDS_SOURCES
    device_context.cpp
)

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

