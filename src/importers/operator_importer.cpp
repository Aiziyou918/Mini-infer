#ifdef MINI_INFER_ONNX_ENABLED

#include "importers/internal/operator_importer.h"
#include "importers/internal/builtin_operators.h"
#include "mini_infer/utils/logger.h"

namespace mini_infer {
namespace importers {

// =============================================================================
// ImporterContext Implementation
// =============================================================================

ImporterContext::ImporterContext(graph::Graph* graph)
    : graph_(graph)
    , verbose_(false) {
}

void ImporterContext::register_tensor(const std::string& name, std::shared_ptr<core::Tensor> tensor) {
    tensors_[name] = tensor;
}

std::shared_ptr<core::Tensor> ImporterContext::get_tensor(const std::string& name) {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) {
        return it->second;
    }
    return nullptr;
}

bool ImporterContext::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

void ImporterContext::register_tensor_producer(const std::string& tensor_name,
                                               const std::string& node_name,
                                               int output_index) {
    tensor_producers_[tensor_name] = TensorProducer{node_name, output_index};
}

bool ImporterContext::get_tensor_producer(const std::string& tensor_name,
                                          std::string& node_name,
                                          int& output_index) const {
    auto it = tensor_producers_.find(tensor_name);
    if (it == tensor_producers_.end()) {
        return false;
    }
    node_name = it->second.node_name;
    output_index = it->second.output_index;
    return true;
}

void ImporterContext::add_node(std::shared_ptr<graph::Node> node) {
    graph_->add_node(node);
}

void ImporterContext::register_weight(const std::string& name, std::shared_ptr<core::Tensor> weight) {
    weights_[name] = weight;
    // Also register as tensor for easy lookup
    register_tensor(name, weight);
}

std::shared_ptr<core::Tensor> ImporterContext::get_weight(const std::string& name) {
    auto it = weights_.find(name);
    if (it != weights_.end()) {
        return it->second;
    }
    return nullptr;
}

bool ImporterContext::is_weight(const std::string& name) const {
    return weights_.find(name) != weights_.end();
}

void ImporterContext::set_error(const std::string& message) {
    if (error_message_.empty()) {
        error_message_ = message;
    } else {
        error_message_ += "; " + message;
    }
    MI_LOG_ERROR("[ImporterContext] " + message);
}

void ImporterContext::log_info(const std::string& message) {
    if (verbose_) {
        MI_LOG_INFO("[ImporterContext] " + message);
    }
}

void ImporterContext::log_warning(const std::string& message) {
    MI_LOG_WARNING("[ImporterContext] " + message);
}

// =============================================================================
// OperatorRegistry Implementation
// =============================================================================

OperatorRegistry::OperatorRegistry() {
    register_builtin_operators();
}

void OperatorRegistry::register_operator(const std::string& op_type, ImporterFactory factory) {
    importers_[op_type] = factory;
}

std::unique_ptr<OperatorImporter> OperatorRegistry::get_importer(const std::string& op_type) {
    auto it = importers_.find(op_type);
    if (it != importers_.end()) {
        return it->second();
    }
    return nullptr;
}

bool OperatorRegistry::is_supported(const std::string& op_type) const {
    return importers_.find(op_type) != importers_.end();
}

std::vector<std::string> OperatorRegistry::get_supported_operators() const {
    std::vector<std::string> ops;
    ops.reserve(importers_.size());
    for (const auto& pair : importers_) {
        ops.push_back(pair.first);
    }
    return ops;
}

void OperatorRegistry::register_builtin_operators() {
    // Register all builtin operators
    // Call the global function from builtin_operators.cpp
    ::mini_infer::importers::register_builtin_operators(*this);
}

} // namespace importers
} // namespace mini_infer

#endif // MINI_INFER_ONNX_ENABLED
