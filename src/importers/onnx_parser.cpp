#ifdef MINI_INFER_ONNX_ENABLED

#include "mini_infer/importers/onnx_parser.h"
#include "mini_infer/importers/model_importer.h"
#include "mini_infer/importers/operator_importer.h"
#include "mini_infer/utils/logger.h"
#include "onnx.pb.h"

#include <fstream>
#include <sstream>

namespace mini_infer {
namespace importers {

OnnxParser::OnnxParser()
    : verbose_(false) {
    // Create operator registry
    operator_registry_ = std::make_unique<OperatorRegistry>();
    
    // Create model importer
    model_importer_ = std::make_unique<ModelImporter>(operator_registry_.get());
    
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
    model_importer_->set_verbose(verbose_);
    
    // Import model
    auto graph = model_importer_->import_model(model);
    if (!graph) {
        set_error("Failed to import ONNX model: " + model_importer_->get_error());
        return nullptr;
    }
    
    log_info("ONNX model imported successfully");
    return graph;
}

OperatorRegistry& OnnxParser::get_registry() {
    return *operator_registry_;
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
