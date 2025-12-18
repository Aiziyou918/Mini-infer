# Graph module
set(GRAPH_SOURCES
    node.cpp
    graph.cpp
    graph_optimizer.cpp
    fusion_pass.cpp
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

# Interface target that keeps graph passes even in static builds
add_library(mini_infer_graph_all INTERFACE)
target_link_libraries(mini_infer_graph_all
    INTERFACE
        $<$<OR:$<BOOL:${BUILD_SHARED_LIBS}>,$<BOOL:${MSVC}>>:mini_infer_graph>
        $<$<AND:$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>,$<NOT:$<BOOL:${MSVC}>>>:-Wl,--whole-archive>
        $<$<AND:$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>,$<NOT:$<BOOL:${MSVC}>>>:mini_infer_graph>
        $<$<AND:$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>,$<NOT:$<BOOL:${MSVC}>>>:-Wl,--no-whole-archive>
)
if (NOT BUILD_SHARED_LIBS AND MSVC)
    target_link_options(mini_infer_graph_all INTERFACE "/WHOLEARCHIVE:mini_infer_graph")
endif()

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

install(TARGETS mini_infer_graph_all
    EXPORT Mini-InferTargets
)
