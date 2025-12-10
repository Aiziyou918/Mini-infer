/**
 * @file op_type.h
 * @brief Operator type definitions (TensorRT-style hybrid architecture)
 *
 * Architecture:
 * - Built-in operators: Use OpType enum (fast switch/case)
 * - Custom operators: Use string (extensibility)
 * - Node caches OpType for performance
 *
 * Inspired by: TensorRT's LayerType enum + IPluginV2::getPluginType()
 */

#pragma once

#include <string>
#include <unordered_map>

namespace mini_infer {
namespace core {

/**
 * @brief Operator Type Enumeration
 *
 * Defines all built-in operator types supported by Mini-Infer.
 * Similar to TensorRT's nvinfer1::LayerType.
 *
 * For custom operators, use OpType::kCUSTOM and check the string name.
 */
enum class OpType {
    // ========================================================================
    // Convolution Operations
    // ========================================================================
    kCONVOLUTION,     ///< 2D Convolution (ONNX: "Conv")
    kCONV_TRANSPOSE,  ///< Transposed Convolution (ONNX: "ConvTranspose")

    // ========================================================================
    // Activation Functions
    // ========================================================================
    kACTIVATION,  ///< Generic activation (use ActivationType for specific type)
    kRELU,        ///< ReLU activation (ONNX: "Relu")
    kSIGMOID,     ///< Sigmoid activation (ONNX: "Sigmoid")
    kTANH,        ///< Tanh activation (ONNX: "Tanh")
    kLEAKY_RELU,  ///< Leaky ReLU (ONNX: "LeakyRelu")
    kPRELU,       ///< Parametric ReLU (ONNX: "PRelu")
    kELU,         ///< Exponential Linear Unit (ONNX: "Elu")

    // ========================================================================
    // Pooling Operations
    // ========================================================================
    kPOOLING,              ///< Generic pooling (use PoolingType for specific type)
    kMAX_POOL,             ///< Max Pooling (ONNX: "MaxPool")
    kAVERAGE_POOL,         ///< Average Pooling (ONNX: "AveragePool")
    kGLOBAL_AVERAGE_POOL,  ///< Global Average Pooling (ONNX: "GlobalAveragePool")
    kGLOBAL_MAX_POOL,      ///< Global Max Pooling (ONNX: "GlobalMaxPool")

    // ========================================================================
    // Normalization Operations
    // ========================================================================
    kNORMALIZATION,  ///< Generic normalization
    kBATCH_NORM,     ///< Batch Normalization (ONNX: "BatchNormalization")
    kINSTANCE_NORM,  ///< Instance Normalization (ONNX: "InstanceNormalization")
    kLAYER_NORM,     ///< Layer Normalization (ONNX: "LayerNormalization")
    kLRN,            ///< Local Response Normalization (ONNX: "LRN")

    // ========================================================================
    // Linear Operations
    // ========================================================================
    kGEMM,    ///< General Matrix Multiplication (ONNX: "Gemm")
    kMATMUL,  ///< Matrix Multiplication (ONNX: "MatMul")
    kLINEAR,  ///< Fully Connected Layer (Custom)

    // ========================================================================
    // Tensor Shape Operations
    // ========================================================================
    kRESHAPE,    ///< Reshape tensor (ONNX: "Reshape")
    kFLATTEN,    ///< Flatten tensor (ONNX: "Flatten")
    kTRANSPOSE,  ///< Transpose tensor (ONNX: "Transpose")
    kSQUEEZE,    ///< Squeeze dimensions (ONNX: "Squeeze")
    kUNSQUEEZE,  ///< Unsqueeze dimensions (ONNX: "Unsqueeze")
    kCONCAT,     ///< Concatenate tensors (ONNX: "Concat")
    kSPLIT,      ///< Split tensor (ONNX: "Split")
    kSHUFFLE,    ///< Shuffle/Permute tensor

    // ========================================================================
    // Element-wise Operations
    // ========================================================================
    kELEMENTWISE,  ///< Generic element-wise operation
    kADD,          ///< Element-wise addition (ONNX: "Add")
    kSUB,          ///< Element-wise subtraction (ONNX: "Sub")
    kMUL,          ///< Element-wise multiplication (ONNX: "Mul")
    kDIV,          ///< Element-wise division (ONNX: "Div")

    // ========================================================================
    // Reduction Operations
    // ========================================================================
    kREDUCE,       ///< Generic reduction
    kREDUCE_SUM,   ///< Reduce sum (ONNX: "ReduceSum")
    kREDUCE_MEAN,  ///< Reduce mean (ONNX: "ReduceMean")
    kREDUCE_MAX,   ///< Reduce max (ONNX: "ReduceMax")
    kREDUCE_MIN,   ///< Reduce min (ONNX: "ReduceMin")

    // ========================================================================
    // Special Operations
    // ========================================================================
    kSOFTMAX,  ///< Softmax operation (ONNX: "Softmax")
    kCAST,     ///< Type casting (ONNX: "Cast")
    kPADDING,  ///< Padding operation (ONNX: "Pad")
    kSLICE,    ///< Slice operation (ONNX: "Slice")

    // ========================================================================
    // Custom/Unknown
    // ========================================================================
    kCUSTOM,  ///< Custom operator (user-defined)
    kUNKNOWN  ///< Unknown operator type
};

/**
 * @brief ONNX operator type name constants
 *
 * These constants map to ONNX operator names.
 * Use these instead of string literals to prevent typos.
 */
namespace op_names {

// Convolution
constexpr const char* kConv = "Conv";
constexpr const char* kConvTranspose = "ConvTranspose";

// Activation
constexpr const char* kRelu = "Relu";
constexpr const char* kSigmoid = "Sigmoid";
constexpr const char* kTanh = "Tanh";
constexpr const char* kLeakyRelu = "LeakyRelu";
constexpr const char* kPRelu = "PRelu";
constexpr const char* kElu = "Elu";

// Pooling
constexpr const char* kMaxPool = "MaxPool";
constexpr const char* kAveragePool = "AveragePool";
constexpr const char* kGlobalAveragePool = "GlobalAveragePool";
constexpr const char* kGlobalMaxPool = "GlobalMaxPool";

// Normalization
constexpr const char* kBatchNormalization = "BatchNormalization";
constexpr const char* kInstanceNormalization = "InstanceNormalization";
constexpr const char* kLayerNormalization = "LayerNormalization";
constexpr const char* kLRN = "LRN";

// Linear
constexpr const char* kGemm = "Gemm";
constexpr const char* kMatMul = "MatMul";
constexpr const char* kLinear = "Linear";

// Shape
constexpr const char* kReshape = "Reshape";
constexpr const char* kFlatten = "Flatten";
constexpr const char* kTranspose = "Transpose";
constexpr const char* kSqueeze = "Squeeze";
constexpr const char* kUnsqueeze = "Unsqueeze";
constexpr const char* kConcat = "Concat";
constexpr const char* kSplit = "Split";

// Element-wise
constexpr const char* kAdd = "Add";
constexpr const char* kSub = "Sub";
constexpr const char* kMul = "Mul";
constexpr const char* kDiv = "Div";

// Reduction
constexpr const char* kReduceSum = "ReduceSum";
constexpr const char* kReduceMean = "ReduceMean";
constexpr const char* kReduceMax = "ReduceMax";
constexpr const char* kReduceMin = "ReduceMin";

// Special
constexpr const char* kSoftmax = "Softmax";
constexpr const char* kCast = "Cast";
constexpr const char* kPad = "Pad";
constexpr const char* kSlice = "Slice";

}  // namespace op_names

/**
 * @brief Convert string operator name to OpType enum
 * @param op_name Operator name string (e.g., "Conv", "Relu")
 * @return OpType enum value
 *
 * This function is called once during Node construction to cache the OpType.
 * For unknown operators, returns OpType::kCUSTOM.
 */
OpType string_to_op_type(const std::string& op_name);

/**
 * @brief Convert OpType enum to string name
 * @param op_type OpType enum value
 * @return Operator name string
 */
const char* op_type_to_string(OpType op_type);

/**
 * @brief Check if operator type is a convolution
 * @param op_type OpType enum value
 * @return true if convolution operator
 */
inline bool is_convolution(OpType op_type) {
    return op_type == OpType::kCONVOLUTION || op_type == OpType::kCONV_TRANSPOSE;
}

/**
 * @brief Check if operator type is an activation function
 * @param op_type OpType enum value
 * @return true if activation operator
 */
inline bool is_activation(OpType op_type) {
    return op_type == OpType::kRELU || op_type == OpType::kSIGMOID || op_type == OpType::kTANH ||
           op_type == OpType::kLEAKY_RELU || op_type == OpType::kPRELU || op_type == OpType::kELU ||
           op_type == OpType::kACTIVATION;
}

/**
 * @brief Check if operator type is a pooling operation
 * @param op_type OpType enum value
 * @return true if pooling operator
 */
inline bool is_pooling(OpType op_type) {
    return op_type == OpType::kMAX_POOL || op_type == OpType::kAVERAGE_POOL ||
           op_type == OpType::kGLOBAL_AVERAGE_POOL || op_type == OpType::kGLOBAL_MAX_POOL ||
           op_type == OpType::kPOOLING;
}

/**
 * @brief Check if operator type is a normalization operation
 * @param op_type OpType enum value
 * @return true if normalization operator
 */
inline bool is_normalization(OpType op_type) {
    return op_type == OpType::kBATCH_NORM || op_type == OpType::kINSTANCE_NORM ||
           op_type == OpType::kLAYER_NORM || op_type == OpType::kLRN ||
           op_type == OpType::kNORMALIZATION;
}

/**
 * @brief Check if operator type is element-wise
 * @param op_type OpType enum value
 * @return true if element-wise operator
 */
inline bool is_elementwise(OpType op_type) {
    return op_type == OpType::kADD || op_type == OpType::kSUB || op_type == OpType::kMUL ||
           op_type == OpType::kDIV || op_type == OpType::kELEMENTWISE;
}

}  // namespace core

// Forward declaration for ActivationType
namespace operators {
enum class ActivationType;
}

namespace core {

/**
 * @brief Convert OpType to ActivationType
 * @param op_type OpType enum value
 * @param[out] act_type Output ActivationType
 * @return true if conversion successful, false if not a supported activation
 *
 * This function provides a fast path for OpType -> ActivationType conversion,
 * avoiding string comparisons in fusion passes.
 *
 * Example:
 *   operators::ActivationType act_type;
 *   if (op_type_to_activation_type(node->type(), act_type)) {
 *       conv->set_activation(act_type);
 *   }
 */
bool op_type_to_activation_type(OpType op_type, operators::ActivationType& act_type);

}  // namespace core
}  // namespace mini_infer
