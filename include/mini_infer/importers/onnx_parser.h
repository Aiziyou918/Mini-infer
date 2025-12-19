#pragma once

#include "mini_infer/graph/graph.h"
#include <string>
#include <memory>
 

namespace mini_infer {
namespace importers {

/**
 * @brief ONNX Parser - Main interface for importing ONNX models
 * 
 * Design philosophy inspired by TensorRT's ONNX parser:
 * - Modular operator importers
 * - Extensible registration system
 * - Robust error handling
 * - Efficient weight management
 */
class OnnxParser {
public:
    OnnxParser();
    ~OnnxParser();

    // Disable copy, allow move
    OnnxParser(const OnnxParser&) = delete;
    OnnxParser& operator=(const OnnxParser&) = delete;
    OnnxParser(OnnxParser&&) noexcept = default;
    OnnxParser& operator=(OnnxParser&&) noexcept = default;

    /**
     * @brief Parse ONNX model from file
     * @param model_path Path to .onnx file
     * @return Parsed graph, nullptr on failure
     */
    std::unique_ptr<graph::Graph> parse_from_file(const std::string& model_path);

    /**
     * @brief Parse ONNX model from memory buffer
     * @param buffer Model data buffer
     * @param size Buffer size in bytes
     * @return Parsed graph, nullptr on failure
     */
    std::unique_ptr<graph::Graph> parse_from_buffer(const void* buffer, size_t size);

    /**
     * @brief Get parser error message
     * @return Error message string
     */
    const std::string& get_error() const { return error_message_; }

    /**
     * @brief Check if parser has errors
     * @return true if has errors
     */
    bool has_error() const { return !error_message_.empty(); }

    /**
     * @brief Enable/disable verbose logging
     * @param verbose true to enable
     */
    void set_verbose(bool verbose) { verbose_ = verbose; }

    /**
     * @brief Get verbose logging state
     * @return true if verbose is enabled
     */
    bool is_verbose() const { return verbose_; }

private:
    std::string error_message_;
    bool verbose_;
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void set_error(const std::string& message);
    void log_info(const std::string& message);
    void log_warning(const std::string& message);
};

} // namespace importers
} // namespace mini_infer
