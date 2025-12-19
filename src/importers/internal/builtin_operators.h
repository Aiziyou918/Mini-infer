#pragma once

#include "importers/internal/operator_importer.h"

namespace mini_infer {
namespace importers {

/**
 * @brief Conv operator importer
 * 
 * ONNX Conv operator:
 * - Inputs: X, W, [B]
 * - Outputs: Y
 * - Attributes: kernel_shape, strides, pads, dilations, group, auto_pad
 */
class ConvImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Conv"; }
};

/**
 * @brief Gemm operator importer (General Matrix Multiplication)
 * 
 * ONNX Gemm operator:
 * - Inputs: A, B, [C]
 * - Outputs: Y
 * - Attributes: alpha, beta, transA, transB
 * - Formula: Y = alpha * A' * B' + beta * C
 */
class GemmImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Gemm"; }
};

/**
 * @brief MatMul operator importer
 * 
 * ONNX MatMul operator:
 * - Inputs: A, B
 * - Outputs: Y
 * - No attributes
 */
class MatMulImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "MatMul"; }
};

/**
 * @brief Relu operator importer
 * 
 * ONNX Relu operator:
 * - Inputs: X
 * - Outputs: Y
 * - No attributes
 */
class ReluImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Relu"; }
};

/**
 * @brief MaxPool operator importer
 * 
 * ONNX MaxPool operator:
 * - Inputs: X
 * - Outputs: Y, [Indices]
 * - Attributes: kernel_shape, strides, pads, dilations, auto_pad
 */
class MaxPoolImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "MaxPool"; }
};

/**
 * @brief AveragePool operator importer
 * 
 * ONNX AveragePool operator:
 * - Inputs: X
 * - Outputs: Y
 * - Attributes: kernel_shape, strides, pads, auto_pad
 */
class AveragePoolImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "AveragePool"; }
};

/**
 * @brief GlobalAveragePool operator importer
 * 
 * ONNX GlobalAveragePool operator:
 * - Inputs: X
 * - Outputs: Y
 * - No attributes (pool over entire spatial dimensions)
 */
class GlobalAveragePoolImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "GlobalAveragePool"; }
};

/**
 * @brief BatchNormalization operator importer
 * 
 * ONNX BatchNormalization operator:
 * - Inputs: X, scale, B, mean, var
 * - Outputs: Y, [mean], [var], [saved_mean], [saved_var]
 * - Attributes: epsilon, momentum
 */
class BatchNormalizationImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "BatchNormalization"; }
};

/**
 * @brief Add operator importer
 * 
 * ONNX Add operator:
 * - Inputs: A, B
 * - Outputs: C
 * - No attributes (element-wise addition)
 */
class AddImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Add"; }
};

/**
 * @brief Mul operator importer
 * 
 * ONNX Mul operator:
 * - Inputs: A, B
 * - Outputs: C
 * - No attributes (element-wise multiplication)
 */
class MulImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Mul"; }
};

/**
 * @brief Reshape operator importer
 * 
 * ONNX Reshape operator:
 * - Inputs: data, shape
 * - Outputs: reshaped
 * - Attributes: allowzero
 */
class ReshapeImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Reshape"; }
};

/**
 * @brief Flatten operator importer
 * 
 * ONNX Flatten operator:
 * - Inputs: input
 * - Outputs: output
 * - Attributes: axis
 */
class FlattenImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Flatten"; }
};

/**
 * @brief Concat operator importer
 * 
 * ONNX Concat operator:
 * - Inputs: inputs (variadic)
 * - Outputs: concat_result
 * - Attributes: axis
 */
class ConcatImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Concat"; }
};

/**
 * @brief Softmax operator importer
 * 
 * ONNX Softmax operator:
 * - Inputs: input
 * - Outputs: output
 * - Attributes: axis
 */
class SoftmaxImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Softmax"; }
};

/**
 * @brief Constant operator importer
 * 
 * ONNX Constant operator:
 * - Inputs: none
 * - Outputs: output
 * - Attributes: value (TensorProto)
 */
class ConstantImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Constant"; }
};

// Register all builtin operators to registry
void register_builtin_operators(OperatorRegistry& registry);

} // namespace importers
} // namespace mini_infer
