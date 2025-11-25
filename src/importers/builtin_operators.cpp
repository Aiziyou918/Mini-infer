#ifdef MINI_INFER_ONNX_ENABLED

#include "mini_infer/importers/builtin_operators.h"
#include "mini_infer/importers/attribute_helper.h"
#include "mini_infer/utils/logger.h"

namespace mini_infer {
namespace importers {

// =============================================================================
// Conv Operator Importer
// =============================================================================

core::Status ConvImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    // Parse attributes
    auto kernel_shape = attrs.get_ints("kernel_shape");
    auto strides = to_int_vector(attrs.get_ints("strides"));
    auto pads = to_int_vector(attrs.get_ints("pads"));
    auto dilations = to_int_vector(attrs.get_ints("dilations"));
    int64_t group = attrs.get_int("group", 1);
    
    ctx.log_info("Conv operator - kernel: [" + 
                 std::to_string(kernel_shape.size() > 0 ? kernel_shape[0] : 0) + 
                 "x" + std::to_string(kernel_shape.size() > 1 ? kernel_shape[1] : 0) + "]");
    
    // TODO: Create Conv operator and add to graph
    ctx.log_warning("Conv operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Gemm Operator Importer
// =============================================================================

core::Status GemmImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    float alpha = attrs.get_float("alpha", 1.0f);
    float beta = attrs.get_float("beta", 1.0f);
    int64_t transA = attrs.get_int("transA", 0);
    int64_t transB = attrs.get_int("transB", 0);
    
    ctx.log_info("Gemm operator - alpha: " + std::to_string(alpha) + 
                 ", beta: " + std::to_string(beta));
    
    // TODO: Create Gemm operator and add to graph
    ctx.log_warning("Gemm operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// MatMul Operator Importer
// =============================================================================

core::Status MatMulImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("MatMul operator");
    
    // TODO: Create MatMul operator and add to graph
    ctx.log_warning("MatMul operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Relu Operator Importer
// =============================================================================

core::Status ReluImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Relu operator");
    
    // TODO: Create Relu operator and add to graph
    ctx.log_warning("Relu operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// MaxPool Operator Importer
// =============================================================================

core::Status MaxPoolImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    auto kernel_shape = attrs.get_ints("kernel_shape");
    auto strides = to_int_vector(attrs.get_ints("strides"));
    auto pads = to_int_vector(attrs.get_ints("pads"));
    
    ctx.log_info("MaxPool operator - kernel: [" + 
                 std::to_string(kernel_shape.size() > 0 ? kernel_shape[0] : 0) + 
                 "x" + std::to_string(kernel_shape.size() > 1 ? kernel_shape[1] : 0) + "]");
    
    // TODO: Create MaxPool operator and add to graph
    ctx.log_warning("MaxPool operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// AveragePool Operator Importer
// =============================================================================

core::Status AveragePoolImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    auto kernel_shape = attrs.get_ints("kernel_shape");
    auto strides = to_int_vector(attrs.get_ints("strides"));
    auto pads = to_int_vector(attrs.get_ints("pads"));
    
    ctx.log_info("AveragePool operator");
    
    // TODO: Create AveragePool operator and add to graph
    ctx.log_warning("AveragePool operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// GlobalAveragePool Operator Importer
// =============================================================================

core::Status GlobalAveragePoolImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("GlobalAveragePool operator");
    
    // TODO: Create GlobalAveragePool operator and add to graph
    ctx.log_warning("GlobalAveragePool operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// BatchNormalization Operator Importer
// =============================================================================

core::Status BatchNormalizationImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    float epsilon = attrs.get_float("epsilon", 1e-5f);
    float momentum = attrs.get_float("momentum", 0.9f);
    
    ctx.log_info("BatchNormalization operator - epsilon: " + std::to_string(epsilon));
    
    // TODO: Create BatchNormalization operator and add to graph
    ctx.log_warning("BatchNormalization operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Add Operator Importer
// =============================================================================

core::Status AddImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Add operator");
    
    // TODO: Create Add operator and add to graph
    ctx.log_warning("Add operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Mul Operator Importer
// =============================================================================

core::Status MulImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Mul operator");
    
    // TODO: Create Mul operator and add to graph
    ctx.log_warning("Mul operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Reshape Operator Importer
// =============================================================================

core::Status ReshapeImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    int64_t allowzero = attrs.get_int("allowzero", 0);
    
    ctx.log_info("Reshape operator");
    
    // TODO: Create Reshape operator and add to graph
    ctx.log_warning("Reshape operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Flatten Operator Importer
// =============================================================================

core::Status FlattenImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    int64_t axis = attrs.get_int("axis", 1);
    
    ctx.log_info("Flatten operator - axis: " + std::to_string(axis));
    
    // TODO: Create Flatten operator and add to graph
    ctx.log_warning("Flatten operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Concat Operator Importer
// =============================================================================

core::Status ConcatImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    int64_t axis = attrs.get_int("axis");
    
    ctx.log_info("Concat operator - axis: " + std::to_string(axis));
    
    // TODO: Create Concat operator and add to graph
    ctx.log_warning("Concat operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Softmax Operator Importer
// =============================================================================

core::Status SoftmaxImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    int64_t axis = attrs.get_int("axis", -1);
    
    ctx.log_info("Softmax operator - axis: " + std::to_string(axis));
    
    // TODO: Create Softmax operator and add to graph
    ctx.log_warning("Softmax operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Constant Operator Importer
// =============================================================================

core::Status ConstantImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    ctx.log_info("Constant operator");
    
    // Constant nodes contain their value in an attribute
    // For now, we'll just register them as constants
    // The actual tensor value would be in the "value" attribute (TensorProto)
    
    // TODO: Extract the constant value from attributes and register as weight
    ctx.log_warning("Constant operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Register All Builtin Operators
// =============================================================================

void register_builtin_operators(OperatorRegistry& registry) {
    // Convolution operators
    REGISTER_ONNX_OPERATOR("Conv", ConvImporter);
    
    // Linear algebra operators
    REGISTER_ONNX_OPERATOR("Gemm", GemmImporter);
    REGISTER_ONNX_OPERATOR("MatMul", MatMulImporter);
    
    // Activation operators
    REGISTER_ONNX_OPERATOR("Relu", ReluImporter);
    
    // Pooling operators
    REGISTER_ONNX_OPERATOR("MaxPool", MaxPoolImporter);
    REGISTER_ONNX_OPERATOR("AveragePool", AveragePoolImporter);
    REGISTER_ONNX_OPERATOR("GlobalAveragePool", GlobalAveragePoolImporter);
    
    // Normalization operators
    REGISTER_ONNX_OPERATOR("BatchNormalization", BatchNormalizationImporter);
    
    // Element-wise operators
    REGISTER_ONNX_OPERATOR("Add", AddImporter);
    REGISTER_ONNX_OPERATOR("Mul", MulImporter);
    
    // Shape operators
    REGISTER_ONNX_OPERATOR("Reshape", ReshapeImporter);
    REGISTER_ONNX_OPERATOR("Flatten", FlattenImporter);
    REGISTER_ONNX_OPERATOR("Concat", ConcatImporter);
    
    // Other operators
    REGISTER_ONNX_OPERATOR("Softmax", SoftmaxImporter);
    REGISTER_ONNX_OPERATOR("Constant", ConstantImporter);
    
    MI_LOG_INFO("[BuiltinOperators] Registered 15 builtin ONNX operators");
}

} // namespace importers
} // namespace mini_infer

#endif // MINI_INFER_ONNX_ENABLED
