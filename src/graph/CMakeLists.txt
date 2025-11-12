# Graph module
set(GRAPH_SOURCES
    node.cpp
    graph.cpp
)

add_library(mini_infer_graph ${GRAPH_SOURCES})

target_include_directories(mini_infer_graph
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mini_infer_graph
    PUBLIC
        mini_infer_core
        mini_infer_operators
)

# Set library properties
set_target_properties(mini_infer_graph PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "mini_infer_graph"
)

# Install
install(TARGETS mini_infer_graph
    EXPORT Mini-InferTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

