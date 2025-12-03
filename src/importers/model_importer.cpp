#ifdef MINI_INFER_ONNX_ENABLED

#include "mini_infer/importers/model_importer.h"
#include "mini_infer/importers/weight_importer.h"
#include "mini_infer/utils/logger.h"
#include "onnx.pb.h"

namespace mini_infer {
namespace importers {

ModelImporter::ModelImporter(OperatorRegistry* registry)
    : registry_(registry)
    , verbose_(false) {
}

const std::string& ModelImporter::get_error() const {
    static std::string empty;
    return empty;
}

std::unique_ptr<graph::Graph> ModelImporter::import_model(const onnx::ModelProto& model) {
    // Log model information
    log_model_info(model);
    
    // Check if model has graph
    if (!model.has_graph()) {
        MI_LOG_ERROR("[ModelImporter] Model has no graph");
        return nullptr;
    }
    
    // Create new graph
    auto graph = std::make_unique<graph::Graph>();
    
    // Create importer context
    ImporterContext ctx(graph.get());
    ctx.set_verbose(verbose_);
    
    // Import graph
    auto status = import_graph(model.graph(), ctx);
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("[ModelImporter] Failed to import graph: " + ctx.get_error());
        return nullptr;
    }
    
    MI_LOG_INFO("[ModelImporter] Model imported successfully");
    return graph;
}

core::Status ModelImporter::import_graph(
    const onnx::GraphProto& graph_proto,
    ImporterContext& ctx
) {
    log_graph_info(graph_proto);
    
    // 1. Import initializers (weights)
    auto status = import_initializers(graph_proto, ctx);
    if (status != core::Status::SUCCESS) {
        return status;
    }
    
    // 2. Import inputs
    status = import_inputs(graph_proto, ctx);
    if (status != core::Status::SUCCESS) {
        return status;
    }
    
    // 3. Import outputs
    status = import_outputs(graph_proto, ctx);
    if (status != core::Status::SUCCESS) {
        return status;
    }
    
    // 4. Import nodes (operators)
    for (int i = 0; i < graph_proto.node_size(); ++i) {
        const auto& node = graph_proto.node(i);
        ctx.log_info("Importing node [" + std::to_string(i) + "]: " + 
                     node.op_type() + " (" + node.name() + ")");
        
        status = import_node(node, ctx);
        if (status != core::Status::SUCCESS) {
            ctx.set_error("Failed to import node: " + node.name());
            return status;
        }
    }


    // Set graph inputs and outputs
    {
        std::vector<std::string> input_names;
        input_names.reserve(graph_proto.input_size());
        for (int i = 0; i < graph_proto.input_size(); ++i) {
            input_names.push_back(graph_proto.input(i).name());
        }
        ctx.get_graph()->set_inputs(input_names);
        
        std::vector<std::string> output_names;
        output_names.reserve(graph_proto.output_size());
        for (int i = 0; i < graph_proto.output_size(); ++i) {
            output_names.push_back(graph_proto.output(i).name());
        }
        ctx.get_graph()->set_outputs(output_names);
    }
    
    return core::Status::SUCCESS;
}

core::Status ModelImporter::import_initializers(
    const onnx::GraphProto& graph_proto,
    ImporterContext& ctx
) {
    ctx.log_info("Importing " + std::to_string(graph_proto.initializer_size()) + " initializers");
    
    for (int i = 0; i < graph_proto.initializer_size(); ++i) {
        const auto& tensor_proto = graph_proto.initializer(i);
        const std::string& name = tensor_proto.name();
        
        // Import tensor using WeightImporter
        std::string error_msg;
        auto tensor = WeightImporter::import_tensor(tensor_proto, error_msg);
        
        if (!tensor) {
            ctx.set_error("  Failed to import initializer '" + name + "': " + error_msg);
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
        
        // Register as weight
        ctx.register_weight(name, tensor);
        ctx.log_info("  Registered weight: " + name + " " + tensor->shape().to_string());
    }
    
    return core::Status::SUCCESS;
}

core::Status ModelImporter::import_inputs(
    const onnx::GraphProto& graph_proto,
    ImporterContext& ctx
) {
    ctx.log_info("Importing " + std::to_string(graph_proto.input_size()) + " inputs");
    
    for (int i = 0; i < graph_proto.input_size(); ++i) {
        const auto& input = graph_proto.input(i);
        const std::string& name = input.name();
        
        // Skip if already registered as weight
        if (ctx.is_weight(name)) {
            ctx.log_info("  Skipping input (is weight): " + name);
            continue;
        }
        
        // Create placeholder tensor for input if not already registered
        if (!ctx.has_tensor(name)) {
            auto input_tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(name, input_tensor);
            ctx.log_info("  Created input placeholder: " + name);
        } else {
            ctx.log_info("  Input already registered: " + name);
        }
    }
    
    return core::Status::SUCCESS;
}

core::Status ModelImporter::import_outputs(
    const onnx::GraphProto& graph_proto,
    ImporterContext& ctx
) {
    ctx.log_info("Importing " + std::to_string(graph_proto.output_size()) + " outputs");
    
    for (int i = 0; i < graph_proto.output_size(); ++i) {
        const auto& output = graph_proto.output(i);
        const std::string& output_name = output.name();
        ctx.log_info("  Output: " + output_name);
        
        // Ensure output tensor exists
        if (!ctx.has_tensor(output_name)) {
            auto output_tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(output_name, output_tensor);
            ctx.log_info("  Created output tensor: " + output_name);
        }
        
        // Mark as graph output (store output names for later use)
        // The actual graph output marking will be done after all nodes are created
    }
    
    return core::Status::SUCCESS;
}

core::Status ModelImporter::import_node(
    const onnx::NodeProto& node,
    ImporterContext& ctx
) {
    const std::string& op_type = node.op_type();
    
    // Check if operator is supported
    if (!registry_->is_supported(op_type)) {
        ctx.log_warning("Unsupported operator: " + op_type);
        return core::Status::ERROR_NOT_IMPLEMENTED;
    }
    
    // Get operator importer
    auto importer = registry_->get_importer(op_type);
    if (!importer) {
        ctx.set_error("Failed to get importer for operator: " + op_type);
        return core::Status::ERROR_RUNTIME;
    }
    
    // Import operator
    return importer->import_operator(ctx, node);
}

void ModelImporter::log_model_info(const onnx::ModelProto& model) {
    if (!verbose_) return;
    
    MI_LOG_INFO("[ModelImporter] ========================================");
    MI_LOG_INFO("[ModelImporter] ONNX Model Information");
    MI_LOG_INFO("[ModelImporter] ========================================");
    MI_LOG_INFO("[ModelImporter] IR Version: " + std::to_string(model.ir_version()));
    
    if (!model.producer_name().empty()) {
        MI_LOG_INFO("[ModelImporter] Producer: " + model.producer_name() + 
                    " " + model.producer_version());
    }
    
    if (!model.domain().empty()) {
        MI_LOG_INFO("[ModelImporter] Domain: " + model.domain());
    }
    
    if (!model.doc_string().empty()) {
        MI_LOG_INFO("[ModelImporter] Description: " + model.doc_string());
    }
}

void ModelImporter::log_graph_info(const onnx::GraphProto& graph_proto) {
    if (!verbose_) return;
    
    MI_LOG_INFO("[ModelImporter] ========================================");
    MI_LOG_INFO("[ModelImporter] Graph: " + graph_proto.name());
    MI_LOG_INFO("[ModelImporter] ========================================");
    MI_LOG_INFO("[ModelImporter] Nodes: " + std::to_string(graph_proto.node_size()));
    MI_LOG_INFO("[ModelImporter] Initializers: " + std::to_string(graph_proto.initializer_size()));
    MI_LOG_INFO("[ModelImporter] Inputs: " + std::to_string(graph_proto.input_size()));
    MI_LOG_INFO("[ModelImporter] Outputs: " + std::to_string(graph_proto.output_size()));
}

} // namespace importers
} // namespace mini_infer

#endif // MINI_INFER_ONNX_ENABLED
