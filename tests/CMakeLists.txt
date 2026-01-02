# Tests using Google Test

set(TEST_SOURCES
    test_tensor.cpp
    test_graph.cpp
    test_operators.cpp
    test_linear.cpp
    test_conv2d.cpp
    test_shape_inference.cpp
    test_optimization_profile.cpp
    test_engine_with_profile.cpp
    test_shape_inference_engine.cpp
    test_dynamic_shape_runtime.cpp
    test_softmax.cpp
    test_pooling.cpp
    test_flatten.cpp
    test_reshape.cpp
    test_kernel_registry.cpp
)

# Add CUDA tests if CUDA is enabled
if(MINI_INFER_ENABLE_CUDA AND CUDAToolkit_FOUND)
    list(APPEND TEST_SOURCES
        test_cuda_kernels.cpp
    )
endif()

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    
    add_executable(${test_name} ${test_src})
    
    target_link_libraries(${test_name}
        PRIVATE
            mini_infer_core
            mini_infer_backends
            mini_infer_operators
            mini_infer_graph_all
            mini_infer_runtime
            mini_infer_kernels_all
            mini_infer_utils
            GTest::gtest_main
    )
    
    # Discover tests automatically with increased timeout
    include(GoogleTest)
    gtest_discover_tests(${test_name})
endforeach()
