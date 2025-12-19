#ifdef MINI_INFER_ONNX_ENABLED

#include "mini_infer/importers/onnx_parser.h"
#include "mini_infer/utils/logger.h"

#include <iostream>
#include <memory>

using namespace mini_infer;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>" << std::endl;
        return 1;
    }
    
    const std::string model_path = argv[1];
    
    // Create ONNX parser
    importers::OnnxParser parser;
    parser.set_verbose(true);
    
    // Parse ONNX model
    MI_LOG_INFO("Parsing ONNX model: " + model_path);
    auto graph = parser.parse_from_file(model_path);
    
    if (!graph) {
        MI_LOG_ERROR("Failed to parse ONNX model: " + parser.get_error());
        return 1;
    }
    
    MI_LOG_INFO("ONNX model parsed successfully!");
    MI_LOG_INFO("Graph nodes: " + std::to_string(graph->nodes().size()));
    
    // Print node information
    const auto& nodes = graph->nodes();
    size_t i = 0;
    for (const auto& pair : nodes) {
        const auto& node = pair.second;
        MI_LOG_INFO("Node[" + std::to_string(i++) + "]: " + node->name());
        // Note: node->get_operator() may be null during import
    }
    
    return 0;
}

#else

#include <iostream>

int main() {
    std::cout << "ONNX support is not enabled. Please build with MINI_INFER_ENABLE_ONNX=ON" << std::endl;
    return 1;
}

#endif // MINI_INFER_ONNX_ENABLED
