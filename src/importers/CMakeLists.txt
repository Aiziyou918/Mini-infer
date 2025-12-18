# ==============================================================================
# Importers Module (ONNX, etc.)
# ==============================================================================

# ONNX is auto-configured by main CMakeLists.txt
# ONNX_PROTO_SOURCES and ONNX_INCLUDE_DIR are set by configure_onnx()

# Source files - New architecture inspired by TensorRT
set(IMPORTERS_SOURCES
    # Main parser interface
    onnx_parser.cpp
    
    # Core importer components
    operator_importer.cpp
    attribute_helper.cpp
    weight_importer.cpp
    model_importer.cpp
    
    # Builtin operator importers
    builtin_operators.cpp
    
    # ONNX Protobuf generated files
    ${ONNX_PROTO_SOURCES}
)

# Create importers library
add_library(mini_infer_importers STATIC ${IMPORTERS_SOURCES})

# Link libraries
target_link_libraries(mini_infer_importers
    PUBLIC
        mini_infer_core
        mini_infer_graph_all
        mini_infer_operators
        mini_infer_utils
        protobuf::libprotobuf
        # Note: Abseil dependencies are automatically handled by protobuf::libprotobuf in MSVC
)

# Include directories
target_include_directories(mini_infer_importers
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${ONNX_INCLUDE_DIR}
    # Note: Protobuf include paths are handled by protobuf::libprotobuf target
)

# Compiler definitions
target_compile_definitions(mini_infer_importers
    PRIVATE
        MINI_INFER_ONNX_ENABLED
)

# Disable protobuf warnings
if(MSVC)
    target_compile_options(mini_infer_importers PRIVATE /wd4244 /wd4267)
else()
    target_compile_options(mini_infer_importers PRIVATE -Wno-unused-parameter)
endif()

message(STATUS "Importers module configured with ONNX support (TensorRT-inspired architecture)")
