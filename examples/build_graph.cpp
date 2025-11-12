#include "mini_infer/graph/graph.h"
#include "mini_infer/runtime/engine.h"
#include "mini_infer/utils/logger.h"
#include <iostream>

using namespace mini_infer;

int main() {
    std::cout << "\n=== Mini-Infer Build Graph Example ===" << std::endl;
    
    // 设置日志级别
    utils::Logger::get_instance().set_level(utils::LogLevel::INFO);
    
    MI_LOG_INFO("Creating graph...");
    auto graph = std::make_shared<graph::Graph>();
    
    // 创建节点
    auto input_node = graph->create_node("input");
    auto conv1_node = graph->create_node("conv1");
    auto conv2_node = graph->create_node("conv2");
    auto output_node = graph->create_node("output");
    
    // 连接节点
    MI_LOG_INFO("Connecting nodes...");
    graph->connect("input", "conv1");
    graph->connect("conv1", "conv2");
    graph->connect("conv2", "output");
    
    // 设置输入输出
    graph->set_inputs({"input"});
    graph->set_outputs({"output"});
    
    // 验证图
    MI_LOG_INFO("Validating graph...");
    auto status = graph->validate();
    if (status == core::Status::SUCCESS) {
        std::cout << "✓ Graph is valid" << std::endl;
    } else {
        std::cout << "✗ Graph validation failed" << std::endl;
        return 1;
    }
    
    // 拓扑排序
    MI_LOG_INFO("Performing topological sort...");
    std::vector<std::shared_ptr<graph::Node>> sorted_nodes;
    status = graph->topological_sort(sorted_nodes);
    
    if (status == core::Status::SUCCESS) {
        std::cout << "Execution order: ";
        for (const auto& node : sorted_nodes) {
            std::cout << node->name() << " -> ";
        }
        std::cout << "done" << std::endl;
    }
    
    // 创建引擎
    MI_LOG_INFO("Creating inference engine...");
    runtime::EngineConfig config;
    config.device_type = core::DeviceType::CPU;
    config.enable_profiling = true;
    
    runtime::Engine engine(config);
    
    MI_LOG_INFO("Build graph example completed successfully!");
    
    return 0;
}

