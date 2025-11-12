# Utils module
set(UTILS_SOURCES
    logger.cpp
)

add_library(mini_infer_utils ${UTILS_SOURCES})

target_include_directories(mini_infer_utils
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mini_infer_utils
    PUBLIC
        Threads::Threads
)

# Set library properties
set_target_properties(mini_infer_utils PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "mini_infer_utils"
)

# Install
install(TARGETS mini_infer_utils
    EXPORT Mini-InferTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

