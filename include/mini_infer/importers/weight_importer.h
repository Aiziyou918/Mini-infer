#pragma once

#include "mini_infer/core/tensor.h"
#include "mini_infer/core/types.h"

#include <memory>
#include <string>

// Forward declarations
namespace onnx {
    class TensorProto;
}

namespace mini_infer {
namespace importers {

/**
 * @brief Weight Importer - Convert ONNX TensorProto to Mini-Infer Tensor
 * 
 * Similar to TensorRT's weight conversion utilities.
 * Handles all ONNX data types and storage formats.
 */
class WeightImporter {
public:
    WeightImporter() = default;
    ~WeightImporter() = default;

    /**
     * @brief Import ONNX tensor to Mini-Infer tensor
     * @param tensor_proto ONNX tensor protobuf
     * @param error_message Output error message
     * @return Imported tensor, nullptr on failure
     */
    static std::shared_ptr<core::Tensor> import_tensor(
        const onnx::TensorProto& tensor_proto,
        std::string& error_message
    );

    /**
     * @brief Convert ONNX data type to Mini-Infer data type
     * @param onnx_dtype ONNX data type enum value
     * @param error_message Output error message
     * @return Mini-Infer data type, or error
     */
    static core::DataType convert_data_type(
        int onnx_dtype,
        std::string& error_message
    );

    /**
     * @brief Get size of ONNX data type in bytes
     * @param onnx_dtype ONNX data type enum value
     * @return Size in bytes, 0 if unknown
     */
    static size_t get_data_type_size(int onnx_dtype);

private:
    // Helper functions for different data storage formats
    static bool import_raw_data(
        const onnx::TensorProto& tensor_proto,
        void* data,
        size_t expected_size
    );

    static bool import_typed_data(
        const onnx::TensorProto& tensor_proto,
        void* data,
        core::DataType dtype,
        size_t num_elements
    );
};

} // namespace importers
} // namespace mini_infer
