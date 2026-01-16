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
 * @brief Gelu operator importer
 *
 * ONNX Gelu operator:
 * - Inputs: X
 * - Outputs: Y
 * - No attributes
 */
class GeluImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Gelu"; }
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
 * @brief Sub operator importer
 *
 * ONNX Sub operator:
 * - Inputs: A, B
 * - Outputs: C
 * - No attributes (element-wise subtraction)
 */
class SubImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Sub"; }
};

/**
 * @brief Div operator importer
 *
 * ONNX Div operator:
 * - Inputs: A, B
 * - Outputs: C
 * - No attributes (element-wise division)
 */
class DivImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Div"; }
};

/**
 * @brief Transpose operator importer
 *
 * ONNX Transpose operator:
 * - Inputs: data
 * - Outputs: transposed
 * - Attributes: perm
 */
class TransposeImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Transpose"; }
};

/**
 * @brief Squeeze operator importer
 *
 * ONNX Squeeze operator:
 * - Inputs: data, [axes]
 * - Outputs: squeezed
 */
class SqueezeImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Squeeze"; }
};

/**
 * @brief Unsqueeze operator importer
 *
 * ONNX Unsqueeze operator:
 * - Inputs: data, axes
 * - Outputs: expanded
 */
class UnsqueezeImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Unsqueeze"; }
};

/**
 * @brief Slice operator importer
 *
 * ONNX Slice operator:
 * - Inputs: data, starts, ends, [axes], [steps]
 * - Outputs: output
 */
class SliceImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Slice"; }
};

/**
 * @brief Gather operator importer
 *
 * ONNX Gather operator:
 * - Inputs: data, indices
 * - Outputs: output
 * - Attributes: axis
 */
class GatherImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Gather"; }
};

/**
 * @brief Shape operator importer
 *
 * ONNX Shape operator:
 * - Inputs: data
 * - Outputs: shape
 */
class ShapeImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Shape"; }
};

/**
 * @brief Cast operator importer
 *
 * ONNX Cast operator:
 * - Inputs: input
 * - Outputs: output
 * - Attributes: to
 */
class CastImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Cast"; }
};

/**
 * @brief ReduceMean operator importer
 *
 * ONNX ReduceMean operator:
 * - Inputs: data
 * - Outputs: reduced
 * - Attributes: axes, keepdims
 */
class ReduceMeanImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "ReduceMean"; }
};

/**
 * @brief LayerNormalization operator importer
 *
 * ONNX LayerNormalization operator:
 * - Inputs: X, Scale, [B]
 * - Outputs: Y, [Mean], [InvStdDev]
 * - Attributes: axis, epsilon
 */
class LayerNormalizationImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "LayerNormalization"; }
};

/**
 * @brief Pow operator importer
 *
 * ONNX Pow operator:
 * - Inputs: X, Y
 * - Outputs: Z
 */
class PowImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Pow"; }
};

/**
 * @brief Sqrt operator importer
 *
 * ONNX Sqrt operator:
 * - Inputs: X
 * - Outputs: Y
 */
class SqrtImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Sqrt"; }
};

/**
 * @brief Erf operator importer
 *
 * ONNX Erf operator:
 * - Inputs: input
 * - Outputs: output
 */
class ErfImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Erf"; }
};

/**
 * @brief Tanh operator importer
 *
 * ONNX Tanh operator:
 * - Inputs: input
 * - Outputs: output
 */
class TanhImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Tanh"; }
};

/**
 * @brief Neg operator importer
 *
 * ONNX Neg operator:
 * - Inputs: X
 * - Outputs: Y
 */
class NegImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Neg"; }
};

/**
 * @brief Exp operator importer
 *
 * ONNX Exp operator:
 * - Inputs: input
 * - Outputs: output
 */
class ExpImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Exp"; }
};

/**
 * @brief Equal operator importer
 *
 * ONNX Equal operator:
 * - Inputs: A, B
 * - Outputs: C
 */
class EqualImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Equal"; }
};

/**
 * @brief Where operator importer
 *
 * ONNX Where operator:
 * - Inputs: condition, X, Y
 * - Outputs: output
 */
class WhereImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Where"; }
};

/**
 * @brief Expand operator importer
 *
 * ONNX Expand operator:
 * - Inputs: input, shape
 * - Outputs: output
 */
class ExpandImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "Expand"; }
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

/**
 * @brief ConstantOfShape operator importer
 *
 * ONNX ConstantOfShape operator:
 * - Inputs: shape (1D tensor)
 * - Outputs: output
 * - Attributes: value (optional tensor with single element)
 */
class ConstantOfShapeImporter : public OperatorImporter {
public:
    core::Status import_operator(ImporterContext& ctx, const onnx::NodeProto& node) override;
    const char* get_op_type() const override { return "ConstantOfShape"; }
};

// Register all builtin operators to registry
void register_builtin_operators(OperatorRegistry& registry);

} // namespace importers
} // namespace mini_infer
