# Tests

# 简单的测试框架（可以后续改用 Google Test）
set(TEST_SOURCES
    test_tensor.cpp
    test_backend.cpp
    test_graph.cpp
)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    
    add_executable(${test_name} ${test_src})
    
    target_link_libraries(${test_name}
        PRIVATE
            mini_infer_core
            mini_infer_backends
            mini_infer_operators
            mini_infer_graph
            mini_infer_runtime
            mini_infer_utils
    )
    
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

