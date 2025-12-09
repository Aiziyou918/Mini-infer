# Runtime module
set(RUNTIME_SOURCES
    engine.cpp
    memory_planner.cpp
)

add_library(mini_infer_runtime ${RUNTIME_SOURCES})

target_include_directories(mini_infer_runtime
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mini_infer_runtime
    PUBLIC
        mini_infer_core
        mini_infer_backends
        mini_infer_operators
        mini_infer_graph
        mini_infer_utils
)

# Set library properties
set_target_properties(mini_infer_runtime PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "mini_infer_runtime"
)

# Install
install(TARGETS mini_infer_runtime
    EXPORT Mini-InferTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

