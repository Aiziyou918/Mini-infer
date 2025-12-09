# CMakeLists.txt 更新说明

## 需要添加的内容

在 `examples/CMakeLists.txt` 中添加新的可执行文件：

```cmake
# LeNet-5 optimized inference with memory planning
add_executable(lenet5_optimized_with_memory_planning
    lenet5_optimized_with_memory_planning.cpp
)

target_link_libraries(lenet5_optimized_with_memory_planning PRIVATE
    mini_infer_core
    mini_infer_operators
    mini_infer_graph
    mini_infer_runtime
    mini_infer_importers
    mini_infer_utils
)

# Set output directory
set_target_properties(lenet5_optimized_with_memory_planning PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
```

## 完整的 examples/CMakeLists.txt 示例

```cmake
# Examples

# LeNet-5 inference (baseline)
add_executable(lenet5_inference
    lenet5_inference.cpp
)

target_link_libraries(lenet5_inference PRIVATE
    mini_infer_core
    mini_infer_operators
    mini_infer_graph
    mini_infer_runtime
    mini_infer_importers
    mini_infer_utils
)

# LeNet-5 optimized inference (graph optimization)
add_executable(lenet5_optimized_inference
    lenet5_optimized_inference.cpp
)

target_link_libraries(lenet5_optimized_inference PRIVATE
    mini_infer_core
    mini_infer_operators
    mini_infer_graph
    mini_infer_runtime
    mini_infer_importers
    mini_infer_utils
)

# LeNet-5 optimized inference with memory planning (NEW!)
add_executable(lenet5_optimized_with_memory_planning
    lenet5_optimized_with_memory_planning.cpp
)

target_link_libraries(lenet5_optimized_with_memory_planning PRIVATE
    mini_infer_core
    mini_infer_operators
    mini_infer_graph
    mini_infer_runtime
    mini_infer_importers
    mini_infer_utils
)

# Memory planner example
add_executable(memory_planner_example
    memory_planner_example.cpp
)

target_link_libraries(memory_planner_example PRIVATE
    mini_infer_core
    mini_infer_operators
    mini_infer_graph
    mini_infer_runtime
    mini_infer_importers
    mini_infer_utils
)

# Set output directory for all examples
set_target_properties(
    lenet5_inference
    lenet5_optimized_inference
    lenet5_optimized_with_memory_planning
    memory_planner_example
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
```

## 验证

编译后，你应该能在 `build/bin/` 目录下看到：

```
build/bin/
├── lenet5_inference.exe
├── lenet5_optimized_inference.exe
├── lenet5_optimized_with_memory_planning.exe  ← 新增
└── memory_planner_example.exe                 ← 新增
```

## 编译命令

```bash
cd build
cmake --build . --config Debug
```

或者只编译新的可执行文件：

```bash
cmake --build . --config Debug --target lenet5_optimized_with_memory_planning
```

## 运行测试

```bash
cd models\python\lenet5
test_optimized_with_memory.bat
```
