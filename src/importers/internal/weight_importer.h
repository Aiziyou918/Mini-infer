#pragma once

#include "mini_infer/core/tensor.h"
#include "mini_infer/core/types.h"

#include <string>
#include <memory>

// Forward declarations
namespace onnx {
    class TensorProto;
}

namespace mini_infer {
namespace importers {

/**
 * @brief Weight Importer - Utilities for importing ONNX weight tensors
 * 
 * Handles conversion from ONNX TensorProto to Mini-Infer Tensor
 */
class WeightImporter {
public:
    /**
     * @brief Import ONNX tensor as weight
     * @param tensor_proto ONNX tensor protobuf
     * @param error_message Output error message
     * @return Shared pointer to Tensor, nullptr on failure
     */
    static std::shared_ptr<core::Tensor> import_tensor(
        const onnx::TensorProto& tensor_proto,
        std::string& error_message
    );

    /**
     * @brief Convert ONNX data type to Mini-Infer data type
     * @param onnx_dtype ONNX data type enum value
     * @param error_message Output error message
     * @return Mini-Infer data type
     */
    static core::DataType convert_data_type(
        int onnx_dtype,
        std::string& error_message
    );

    /**
     * @brief Get data type size in bytes
     * @param onnx_dtype ONNX data type enum value
     * @return Size in bytes
     */
    static size_t get_data_type_size(int onnx_dtype);

private:
    static bool import_raw_data(
        const onnx::TensorProto& tensor_proto,
        void* data_buffer,
        size_t data_size,
        std::string& error_message
    );

    static bool import_typed_data(
        const onnx::TensorProto& tensor_proto,
        void* data_buffer,
        size_t data_size,
        std::string& error_message
    );
};

} // namespace importers
} // namespace mini_infer
