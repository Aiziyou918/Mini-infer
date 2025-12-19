#pragma once

#include "importers/internal/operator_importer.h"
#include "mini_infer/graph/graph.h"
#include "mini_infer/core/types.h"

#include <string>
#include <memory>

// Forward declarations
namespace onnx {
    class ModelProto;
    class GraphProto;
    class NodeProto;
}

namespace mini_infer {
namespace importers {

/**
 * @brief Model Importer - Main ONNX model import logic
 * 
 * Similar to TensorRT's ModelImporter.
 * Orchestrates the import process:
 * 1. Parse ONNX ModelProto
 * 2. Import weights (initializers)
 * 3. Import nodes (operators)
 * 4. Build graph connections
 */
class ModelImporter {
public:
    explicit ModelImporter(OperatorRegistry* registry);
    ~ModelImporter() = default;

    /**
     * @brief Import ONNX model from protobuf
     * @param model ONNX ModelProto
     * @return Imported graph, nullptr on failure
     */
    std::unique_ptr<graph::Graph> import_model(const onnx::ModelProto& model);

    /**
     * @brief Get last error message
     * @return Error message string
     */
    const std::string& get_error() const;

    /**
     * @brief Enable/disable verbose logging
     * @param verbose true to enable
     */
    void set_verbose(bool verbose) { verbose_ = verbose; }

private:
    OperatorRegistry* registry_;
    bool verbose_;

    /**
     * @brief Import ONNX GraphProto
     * @param graph_proto ONNX graph
     * @return Status code
     */
    core::Status import_graph(
        const onnx::GraphProto& graph_proto,
        ImporterContext& ctx
    );

    /**
     * @brief Import initializers (weights)
     * @param graph_proto ONNX graph
     * @param ctx Importer context
     * @return Status code
     */
    core::Status import_initializers(
        const onnx::GraphProto& graph_proto,
        ImporterContext& ctx
    );

    /**
     * @brief Import graph inputs
     * @param graph_proto ONNX graph
     * @param ctx Importer context
     * @return Status code
     */
    core::Status import_inputs(
        const onnx::GraphProto& graph_proto,
        ImporterContext& ctx
    );

    /**
     * @brief Import graph outputs
     * @param graph_proto ONNX graph
     * @param ctx Importer context
     * @return Status code
     */
    core::Status import_outputs(
        const onnx::GraphProto& graph_proto,
        ImporterContext& ctx
    );

    /**
     * @brief Import a single operator node
     * @param node ONNX node
     * @param ctx Importer context
     * @return Status code
     */
    core::Status import_node(
        const onnx::NodeProto& node,
        ImporterContext& ctx
    );

    /**
     * @brief Log model information
     * @param model ONNX model
     */
    void log_model_info(const onnx::ModelProto& model);

    /**
     * @brief Log graph information
     * @param graph_proto ONNX graph
     */
    void log_graph_info(const onnx::GraphProto& graph_proto);
};

} // namespace importers
} // namespace mini_infer
