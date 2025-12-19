#ifdef MINI_INFER_ONNX_ENABLED

#include "importers/internal/weight_importer.h"
#include "onnx.pb.h"

#include <cstring>

namespace mini_infer {
namespace importers {

std::shared_ptr<core::Tensor> WeightImporter::import_tensor(
    const onnx::TensorProto& tensor_proto,
    std::string& error_message
) {
    // 1. Extract shape
    std::vector<int64_t> dims;
    for (int i = 0; i < tensor_proto.dims_size(); ++i) {
        dims.push_back(tensor_proto.dims(i));
    }
    core::Shape shape(dims);
    
    // 2. Convert data type
    core::DataType dtype = convert_data_type(tensor_proto.data_type(), error_message);
    if (!error_message.empty()) {
        return nullptr;
    }
    
    // 3. Create tensor
    auto tensor = std::make_shared<core::Tensor>(shape, dtype);
    
    // 4. Import data
    void* data = tensor->data();
    size_t num_elements = tensor->shape().numel();
    
    if (tensor_proto.has_raw_data()) {
        // Raw binary data (most efficient)
        size_t expected_size = num_elements * tensor->element_size();
        if (!import_raw_data(tensor_proto, data, expected_size, error_message)) {
            return nullptr;
        }
    } else {
        // Typed data arrays
        size_t expected_size = num_elements * tensor->element_size();
        if (!import_typed_data(tensor_proto, data, expected_size, error_message)) {
            return nullptr;
        }
    }
    
    return tensor;
}

core::DataType WeightImporter::convert_data_type(int onnx_dtype, std::string& error_message) {
    switch (onnx_dtype) {
        case onnx::TensorProto::FLOAT:
            return core::DataType::FLOAT32;
        case onnx::TensorProto::FLOAT16:
            return core::DataType::FLOAT16;
        case onnx::TensorProto::INT32:
            return core::DataType::INT32;
        case onnx::TensorProto::INT64:
            return core::DataType::INT64;
        case onnx::TensorProto::INT8:
            return core::DataType::INT8;
        case onnx::TensorProto::UINT8:
            return core::DataType::UINT8;
        case onnx::TensorProto::BOOL:
            return core::DataType::BOOL;
        default:
            error_message = "Unsupported ONNX data type: " + std::to_string(onnx_dtype);
            return core::DataType::FLOAT32; // Return default on error
    }
}

size_t WeightImporter::get_data_type_size(int onnx_dtype) {
    switch (onnx_dtype) {
        case onnx::TensorProto::FLOAT: return 4;
        case onnx::TensorProto::FLOAT16: return 2;
        case onnx::TensorProto::INT32: return 4;
        case onnx::TensorProto::INT64: return 8;
        case onnx::TensorProto::INT8: return 1;
        case onnx::TensorProto::UINT8: return 1;
        case onnx::TensorProto::BOOL: return 1;
        default: return 0;
    }
}

bool WeightImporter::import_raw_data(
    const onnx::TensorProto& tensor_proto,
    void* data,
    size_t expected_size,
    std::string& error_message
) {
    const std::string& raw = tensor_proto.raw_data();
    if (raw.size() != expected_size) {
        error_message = "Raw data size mismatch: expected " + std::to_string(expected_size) +
                        ", got " + std::to_string(raw.size());
        return false;
    }
    std::memcpy(data, raw.data(), raw.size());
    return true;
}

bool WeightImporter::import_typed_data(
    const onnx::TensorProto& tensor_proto,
    void* data,
    size_t data_size,
    std::string& error_message
) {
    const int onnx_dtype = tensor_proto.data_type();
    const size_t element_size = get_data_type_size(onnx_dtype);
    if (element_size == 0) {
        error_message = "Unsupported ONNX data type: " + std::to_string(onnx_dtype);
        return false;
    }
    if (data_size % element_size != 0) {
        error_message = "Typed data size is not aligned to element size";
        return false;
    }
    const size_t num_elements = data_size / element_size;

    switch (onnx_dtype) {
        case onnx::TensorProto::FLOAT: {
            if (tensor_proto.float_data_size() != static_cast<int>(num_elements)) {
                error_message = "FLOAT data size mismatch";
                return false;
            }
            float* float_data = static_cast<float*>(data);
            for (int i = 0; i < tensor_proto.float_data_size(); ++i) {
                float_data[i] = tensor_proto.float_data(i);
            }
            return true;
        }
        case onnx::TensorProto::INT32: {
            if (tensor_proto.int32_data_size() != static_cast<int>(num_elements)) {
                error_message = "INT32 data size mismatch";
                return false;
            }
            int32_t* int_data = static_cast<int32_t*>(data);
            for (int i = 0; i < tensor_proto.int32_data_size(); ++i) {
                int_data[i] = tensor_proto.int32_data(i);
            }
            return true;
        }
        case onnx::TensorProto::INT64: {
            if (tensor_proto.int64_data_size() != static_cast<int>(num_elements)) {
                error_message = "INT64 data size mismatch";
                return false;
            }
            int64_t* int64_data = static_cast<int64_t*>(data);
            for (int i = 0; i < tensor_proto.int64_data_size(); ++i) {
                int64_data[i] = tensor_proto.int64_data(i);
            }
            return true;
        }
        default:
            error_message = "Typed data not supported for ONNX data type: " +
                            std::to_string(onnx_dtype);
            return false;
    }
}

} // namespace importers
} // namespace mini_infer

#endif // MINI_INFER_ONNX_ENABLED
