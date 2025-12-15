# Examples

set(EXAMPLE_SOURCES
    build_graph.cpp
    lenet5_inference.cpp
    shape_inference_demo.cpp
    dynamic_shape_basic.cpp
    dynamic_shape_advanced.cpp
)

# ONNX examples (only if ONNX is enabled)
if(MINI_INFER_ENABLE_ONNX)
    list(APPEND EXAMPLE_SOURCES onnx_parser_example.cpp)
    list(APPEND EXAMPLE_SOURCES lenet5_onnx_test.cpp)
    list(APPEND EXAMPLE_SOURCES lenet5_optimized_inference.cpp)
    list(APPEND EXAMPLE_SOURCES lenet5_optimized_with_memory_planning.cpp)
endif()

foreach(example_src ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_src} NAME_WE)
    
    add_executable(${example_name} ${example_src})
    
    target_link_libraries(${example_name}
        PRIVATE
            mini_infer_core
            mini_infer_backends
            mini_infer_operators
            mini_infer_graph
            mini_infer_runtime
            mini_infer_utils
    )
    
    # Link ONNX importers if ONNX is enabled and this is an ONNX example
    if(MINI_INFER_ENABLE_ONNX)
        if(example_name STREQUAL "onnx_parser_example" OR 
           example_name STREQUAL "lenet5_onnx_test" OR
           example_name STREQUAL "lenet5_optimized_inference" OR
           example_name STREQUAL "lenet5_optimized_with_memory_planning")
            target_link_libraries(${example_name} PRIVATE mini_infer_importers)
        endif()
        
    endif()
endforeach()

