cmake_minimum_required(VERSION 3.18)

project(Mini-Infer
    VERSION 0.1.0
    DESCRIPTION "A lightweight inference framework similar to TensorRT"
    LANGUAGES CXX
)

# ==============================================================================
# Project Configuration
# ==============================================================================

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ==============================================================================
# Build Options
# ==============================================================================

option(MINI_INFER_BUILD_TESTS "Build tests" ON)
option(MINI_INFER_BUILD_EXAMPLES "Build examples" ON)
option(MINI_INFER_BUILD_SHARED_LIBS "Build shared libraries" ON)
option(MINI_INFER_ENABLE_CUDA "Enable CUDA backend (future support)" OFF)
option(MINI_INFER_ENABLE_PROFILING "Enable profiling support" ON)
option(MINI_INFER_ENABLE_LOGGING "Enable logging support" ON)

# ==============================================================================
# Global Settings
# ==============================================================================

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Position Independent Code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# ==============================================================================
# Compiler Flags
# ==============================================================================

if(MSVC)
    # MSVC specific flags
    add_compile_options(
        /W4          # Warning level 4
        /WX-         # Warnings not as errors (remove - for strict)
        /MP          # Multi-processor compilation
        /permissive- # Standards conformance
    )
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DNOMINMAX)
else()
    # GCC/Clang flags
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3 -march=native)
    endif()
endif()

# ==============================================================================
# Dependencies
# ==============================================================================

# Find threads (required for most parallel processing)
find_package(Threads REQUIRED)

# Google Test (for unit testing)
if(MINI_INFER_BUILD_TESTS)
    # For Windows MinGW: Configure GTest to use Windows native threading
    if(WIN32 AND MINGW)
        set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
        add_definitions(-DGTEST_HAS_PTHREAD=0)
    endif()
    
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    # Ensure GTest doesn't use pthread on Windows MinGW
    if(WIN32 AND MINGW)
        target_compile_definitions(gtest PUBLIC GTEST_HAS_PTHREAD=0)
        target_compile_definitions(gtest_main PUBLIC GTEST_HAS_PTHREAD=0)
    endif()
endif()

# CUDA support (future)
if(MINI_INFER_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DMINI_INFER_CUDA_ENABLED)
endif()

# ==============================================================================
# Subdirectories
# ==============================================================================

# Core library
add_subdirectory(src/core)

# Backend implementations
add_subdirectory(src/backends)

# Operators
add_subdirectory(src/operators)

# Graph module
add_subdirectory(src/graph)

# Runtime
add_subdirectory(src/runtime)

# Utils
add_subdirectory(src/utils)

# Tests
if(MINI_INFER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(MINI_INFER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ==============================================================================
# Installation
# ==============================================================================

include(GNUInstallDirs)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Export targets
install(EXPORT Mini-InferTargets
    FILE Mini-InferTargets.cmake
    NAMESPACE MiniInfer::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Mini-Infer
)

# ==============================================================================
# Summary
# ==============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Mini-Infer Configuration Summary")
message(STATUS "========================================")
message(STATUS "Version:           ${PROJECT_VERSION}")
message(STATUS "Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Tests:       ${MINI_INFER_BUILD_TESTS}")
message(STATUS "Build Examples:    ${MINI_INFER_BUILD_EXAMPLES}")
message(STATUS "Build Shared Libs: ${MINI_INFER_BUILD_SHARED_LIBS}")
message(STATUS "Enable CUDA:       ${MINI_INFER_ENABLE_CUDA}")
message(STATUS "Enable Profiling:  ${MINI_INFER_ENABLE_PROFILING}")
message(STATUS "Enable Logging:    ${MINI_INFER_ENABLE_LOGGING}")
message(STATUS "========================================")
message(STATUS "")

