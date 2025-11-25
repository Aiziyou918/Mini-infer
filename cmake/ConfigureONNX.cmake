# ==============================================================================
# ONNX Auto Configuration Module
# ==============================================================================
# 
# This module automatically:
# 1. Downloads ONNX proto files if missing
# 2. Finds compatible protoc
# 3. Generates C++ files from proto definitions
# 4. Handles version compatibility
#
# Usage:
#   include(cmake/ConfigureONNX.cmake)
#   configure_onnx()
#

function(configure_onnx)
    message(STATUS "========================================")
    message(STATUS "Configuring ONNX Environment")
    message(STATUS "========================================")
    
    # Set paths
    set(ONNX_DIR "${PROJECT_SOURCE_DIR}/third_party/onnx")
    set(ONNX_PROTO_FILE "${ONNX_DIR}/onnx.proto")
    set(ONNX_PB_CC_FILE "${ONNX_DIR}/onnx.pb.cc")
    set(ONNX_PB_H_FILE "${ONNX_DIR}/onnx.pb.h")
    
    # Create ONNX directory if not exists
    if(NOT EXISTS "${ONNX_DIR}")
        file(MAKE_DIRECTORY "${ONNX_DIR}")
        message(STATUS "Created ONNX directory: ${ONNX_DIR}")
    endif()
    
    # Step 1: Download onnx.proto if missing
    if(NOT EXISTS "${ONNX_PROTO_FILE}")
        message(STATUS "Downloading onnx.proto...")
        set(ONNX_VERSION "v1.15.0")
        set(ONNX_PROTO_URL "https://raw.githubusercontent.com/onnx/onnx/${ONNX_VERSION}/onnx/onnx.proto")
        
        file(DOWNLOAD 
            "${ONNX_PROTO_URL}" 
            "${ONNX_PROTO_FILE}"
            STATUS download_status
            SHOW_PROGRESS
        )
        
        list(GET download_status 0 status_code)
        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "Failed to download onnx.proto from ${ONNX_PROTO_URL}")
        endif()
        
        message(STATUS "[SUCCESS] Downloaded onnx.proto successfully")
    else()
        message(STATUS "[SUCCESS] onnx.proto already exists")
    endif()
    
    # Step 2: Find protoc
    find_program(PROTOC_EXECUTABLE protoc)
    if(NOT PROTOC_EXECUTABLE)
        message(FATAL_ERROR "protoc not found! Please install Protocol Buffers.")
    endif()
    
    # Get protoc version
    execute_process(
        COMMAND ${PROTOC_EXECUTABLE} --version
        OUTPUT_VARIABLE PROTOC_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Found protoc: ${PROTOC_EXECUTABLE}")
    message(STATUS "Protoc version: ${PROTOC_VERSION_OUTPUT}")
    
    # Step 3: Check if we need to regenerate
    set(NEED_REGENERATE FALSE)
    
    if(NOT EXISTS "${ONNX_PB_CC_FILE}" OR NOT EXISTS "${ONNX_PB_H_FILE}")
        set(NEED_REGENERATE TRUE)
        message(STATUS "Generated files missing, will regenerate")
    else()
        # Check if proto file is newer than generated files
        file(TIMESTAMP "${ONNX_PROTO_FILE}" PROTO_TIME)
        file(TIMESTAMP "${ONNX_PB_CC_FILE}" PB_CC_TIME)
        
        if("${PROTO_TIME}" IS_NEWER_THAN "${PB_CC_TIME}")
            set(NEED_REGENERATE TRUE)
            message(STATUS "Proto file is newer, will regenerate")
        endif()
    endif()
    
    # Step 4: Generate C++ files if needed
    if(NEED_REGENERATE)
        message(STATUS "Generating C++ files from onnx.proto...")
        
        # Remove old files first
        file(REMOVE "${ONNX_PB_CC_FILE}" "${ONNX_PB_H_FILE}")
        
        # Generate new files
        execute_process(
            COMMAND ${PROTOC_EXECUTABLE} --cpp_out=${ONNX_DIR} onnx.proto
            WORKING_DIRECTORY ${ONNX_DIR}
            RESULT_VARIABLE protoc_result
            OUTPUT_VARIABLE protoc_output
            ERROR_VARIABLE protoc_error
        )
        
        if(NOT protoc_result EQUAL 0)
            message(FATAL_ERROR "Failed to generate C++ files from onnx.proto:\n${protoc_error}")
        endif()
        
        # Verify generated files
        if(EXISTS "${ONNX_PB_CC_FILE}" AND EXISTS "${ONNX_PB_H_FILE}")
            file(SIZE "${ONNX_PB_CC_FILE}" PB_CC_SIZE)
            file(SIZE "${ONNX_PB_H_FILE}" PB_H_SIZE)
            math(EXPR PB_CC_SIZE_KB "${PB_CC_SIZE} / 1024")
            math(EXPR PB_H_SIZE_KB "${PB_H_SIZE} / 1024")
            
            message(STATUS "[SUCCESS] Generated onnx.pb.cc (${PB_CC_SIZE_KB} KB)")
            message(STATUS "[SUCCESS] Generated onnx.pb.h (${PB_H_SIZE_KB} KB)")
        else()
            message(FATAL_ERROR "Generated files not found after protoc execution")
        endif()
    else()
        message(STATUS "[SUCCESS] Generated files are up to date")
    endif()
    
    # Step 5: Set output variables for parent scope
    set(ONNX_PROTO_SOURCES "${ONNX_PB_CC_FILE}" PARENT_SCOPE)
    set(ONNX_PROTO_HEADERS "${ONNX_PB_H_FILE}" PARENT_SCOPE)
    set(ONNX_INCLUDE_DIR "${ONNX_DIR}" PARENT_SCOPE)
    
    message(STATUS "========================================")
    message(STATUS "ONNX Configuration Complete!")
    message(STATUS "========================================")
    
endfunction()

# Utility function to check ONNX prerequisites
function(check_onnx_prerequisites)
    # Check if Protobuf is found
    if(NOT Protobuf_FOUND)
        message(WARNING "Protobuf not found. ONNX support will be disabled.")
        return()
    endif()
    
    # Check if protoc is available
    find_program(PROTOC_EXECUTABLE protoc)
    if(NOT PROTOC_EXECUTABLE)
        message(WARNING "protoc not found. ONNX support will be disabled.")
        return()
    endif()
    
    set(ONNX_PREREQUISITES_MET TRUE PARENT_SCOPE)
endfunction()
