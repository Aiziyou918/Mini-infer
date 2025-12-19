#ifdef MINI_INFER_ONNX_ENABLED

#include "mini_infer/importers/onnx_parser.h"
#include "importers/internal/model_importer.h"
#include "importers/internal/operator_importer.h"
#include "mini_infer/utils/logger.h"
#include "onnx.pb.h"

#include <fstream>
#include <sstream>

namespace mini_infer {
namespace importers {

struct OnnxParser::Impl {
    std::unique_ptr<OperatorRegistry> operator_registry;
    std::unique_ptr<ModelImporter> model_importer;

    Impl() {
        operator_registry = std::make_unique<OperatorRegistry>();
        model_importer = std::make_unique<ModelImporter>(operator_registry.get());
    }
};

OnnxParser::OnnxParser()
    : verbose_(false), impl_(std::make_unique<Impl>()) {
    MI_LOG_INFO("[OnnxParser] ONNX Parser initialized (TensorRT-inspired architecture)");
}

OnnxParser::~OnnxParser() = default;

std::unique_ptr<graph::Graph> OnnxParser::parse_from_file(const std::string& model_path) {
    log_info("Parsing ONNX model from file: " + model_path);
    
    // Read file
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        set_error("Failed to open file: " + model_path);
        return nullptr;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Read file content
    std::vector<char> buffer(file_size);
    file.read(buffer.data(), file_size);
    file.close();
    
    if (file.gcount() != static_cast<std::streamsize>(file_size)) {
        set_error("Failed to read complete file: " + model_path);
        return nullptr;
    }
    
    log_info("File read successfully, size: " + std::to_string(file_size) + " bytes");
    
    // Parse from buffer
    return parse_from_buffer(buffer.data(), file_size);
}

std::unique_ptr<graph::Graph> OnnxParser::parse_from_buffer(const void* buffer, size_t size) {
    log_info("Parsing ONNX model from buffer, size: " + std::to_string(size) + " bytes");
    
    // Clear previous errors
    error_message_.clear();
    
    // Parse protobuf
    onnx::ModelProto model;
    if (!model.ParseFromArray(buffer, static_cast<int>(size))) {
        set_error("Failed to parse ONNX protobuf");
        return nullptr;
    }
    
    log_info("Protobuf parsed successfully");
    
    // Set verbose mode for model importer
    impl_->model_importer->set_verbose(verbose_);
    
    // Import model
    auto graph = impl_->model_importer->import_model(model);
    if (!graph) {
        set_error("Failed to import ONNX model: " + impl_->model_importer->get_error());
        return nullptr;
    }
    
    log_info("ONNX model imported successfully");
    return graph;
}

void OnnxParser::set_error(const std::string& message) {
    if (error_message_.empty()) {
        error_message_ = message;
    } else {
        error_message_ += "; " + message;
    }
    MI_LOG_ERROR("[OnnxParser] " + message);
}

void OnnxParser::log_info(const std::string& message) {
    if (verbose_) {
        MI_LOG_INFO("[OnnxParser] " + message);
    }
}

void OnnxParser::log_warning(const std::string& message) {
    MI_LOG_WARNING("[OnnxParser] " + message);
}

} // namespace importers
} // namespace mini_infer

#endif // MINI_INFER_ONNX_ENABLED
