#pragma once

#include "mini_infer/graph/graph.h"
#include "mini_infer/graph/node.h"
#include "mini_infer/core/tensor.h"
#include "mini_infer/core/types.h"

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

// Forward declarations
namespace onnx {
    class NodeProto;
    class GraphProto;
    class TensorProto;
}

namespace mini_infer {
namespace importers {

// Forward declarations
class ImporterContext;

/**
 * @brief Operator Importer Base Class
 * 
 * Each ONNX operator type has its own importer that inherits from this class.
 * Similar to TensorRT's builtin_op_importers design.
 */
class OperatorImporter {
public:
    virtual ~OperatorImporter() = default;

    /**
     * @brief Import ONNX operator node to Mini-Infer graph
     * @param ctx Importer context containing graph and tensors
     * @param node ONNX node to import
     * @return Status code
     */
    virtual core::Status import_operator(
        ImporterContext& ctx,
        const onnx::NodeProto& node
    ) = 0;

    /**
     * @brief Get operator type name
     * @return Operator type string
     */
    virtual const char* get_op_type() const = 0;
};

/**
 * @brief Importer Context - Shared state during import
 * 
 * Similar to TensorRT's ImporterContext, provides access to:
 * - Graph being built
 * - Tensor registry
 * - Weight management
 * - Error handling
 */
class ImporterContext {
public:
    ImporterContext(graph::Graph* graph);
    ~ImporterContext() = default;

    // Graph access
    graph::Graph* get_graph() { return graph_; }

    // Tensor management
    void register_tensor(const std::string& name, std::shared_ptr<core::Tensor> tensor);
    std::shared_ptr<core::Tensor> get_tensor(const std::string& name);
    bool has_tensor(const std::string& name) const;

    // Node management
    void add_node(std::shared_ptr<graph::Node> node);

    // Weight management (initializers)
    void register_weight(const std::string& name, std::shared_ptr<core::Tensor> weight);
    std::shared_ptr<core::Tensor> get_weight(const std::string& name);
    bool is_weight(const std::string& name) const;

    // Error handling
    void set_error(const std::string& message);
    const std::string& get_error() const { return error_message_; }
    bool has_error() const { return !error_message_.empty(); }

    // Logging
    void log_info(const std::string& message);
    void log_warning(const std::string& message);
    void set_verbose(bool verbose) { verbose_ = verbose; }

private:
    graph::Graph* graph_;
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> tensors_;
    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> weights_;
    std::string error_message_;
    bool verbose_;
};

/**
 * @brief Operator Registry - Registration system for operator importers
 * 
 * Similar to TensorRT's operator registration mechanism.
 * Allows dynamic registration and lookup of operator importers.
 */
class OperatorRegistry {
public:
    using ImporterFactory = std::function<std::unique_ptr<OperatorImporter>()>;

    OperatorRegistry();
    ~OperatorRegistry() = default;

    /**
     * @brief Register operator importer
     * @param op_type ONNX operator type name
     * @param factory Factory function to create importer
     */
    void register_operator(const std::string& op_type, ImporterFactory factory);

    /**
     * @brief Get operator importer by type
     * @param op_type ONNX operator type name
     * @return Importer instance, nullptr if not found
     */
    std::unique_ptr<OperatorImporter> get_importer(const std::string& op_type);

    /**
     * @brief Check if operator is supported
     * @param op_type ONNX operator type name
     * @return true if supported
     */
    bool is_supported(const std::string& op_type) const;

    /**
     * @brief Get all supported operator types
     * @return Vector of operator type names
     */
    std::vector<std::string> get_supported_operators() const;

private:
    std::unordered_map<std::string, ImporterFactory> importers_;

    void register_builtin_operators();
};

// Helper macro for registering operators
#define REGISTER_ONNX_OPERATOR(op_type, importer_class) \
    registry.register_operator(op_type, []() -> std::unique_ptr<OperatorImporter> { \
        return std::make_unique<importer_class>(); \
    })

} // namespace importers
} // namespace mini_infer
