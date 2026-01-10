/**
 * @file op_type.cpp
 * @brief Operator type conversion functions implementation
 */

#include "mini_infer/core/op_type.h"

#include <unordered_map>


namespace mini_infer {
namespace core {

namespace {

// Static mapping table: String -> OpType
// Initialized once, used for fast lookup
const std::unordered_map<std::string, OpType> kStringToOpTypeMap = {
    // Convolution
    {op_names::kConv, OpType::kCONVOLUTION},
    {op_names::kConvTranspose, OpType::kCONV_TRANSPOSE},

    // Activation
    {op_names::kRelu, OpType::kRELU},
    {op_names::kSigmoid, OpType::kSIGMOID},
    {op_names::kTanh, OpType::kTANH},
    {op_names::kLeakyRelu, OpType::kLEAKY_RELU},
    {op_names::kPRelu, OpType::kPRELU},
    {op_names::kElu, OpType::kELU},
    {op_names::kGelu, OpType::kGELU},

    // Pooling
    {op_names::kMaxPool, OpType::kMAX_POOL},
    {op_names::kAveragePool, OpType::kAVERAGE_POOL},
    {op_names::kGlobalAveragePool, OpType::kGLOBAL_AVERAGE_POOL},
    {op_names::kGlobalMaxPool, OpType::kGLOBAL_MAX_POOL},

    // Normalization
    {op_names::kBatchNormalization, OpType::kBATCH_NORM},
    {op_names::kInstanceNormalization, OpType::kINSTANCE_NORM},
    {op_names::kLayerNormalization, OpType::kLAYER_NORM},
    {op_names::kLRN, OpType::kLRN},

    // Linear
    {op_names::kGemm, OpType::kGEMM},
    {op_names::kMatMul, OpType::kMATMUL},
    {op_names::kLinear, OpType::kLINEAR},

    // Shape
    {op_names::kReshape, OpType::kRESHAPE},
    {op_names::kFlatten, OpType::kFLATTEN},
    {op_names::kTranspose, OpType::kTRANSPOSE},
    {op_names::kSqueeze, OpType::kSQUEEZE},
    {op_names::kUnsqueeze, OpType::kUNSQUEEZE},
    {op_names::kConcat, OpType::kCONCAT},
    {op_names::kSplit, OpType::kSPLIT},
    {op_names::kShape, OpType::kSHAPE},

    // Element-wise
    {op_names::kAdd, OpType::kADD},
    {op_names::kSub, OpType::kSUB},
    {op_names::kMul, OpType::kMUL},
    {op_names::kDiv, OpType::kDIV},

    // Reduction
    {op_names::kReduceSum, OpType::kREDUCE_SUM},
    {op_names::kReduceMean, OpType::kREDUCE_MEAN},
    {op_names::kReduceMax, OpType::kREDUCE_MAX},
    {op_names::kReduceMin, OpType::kREDUCE_MIN},

    // Special
    {op_names::kSoftmax, OpType::kSOFTMAX},
    {op_names::kCast, OpType::kCAST},
    {op_names::kPad, OpType::kPADDING},
    {op_names::kSlice, OpType::kSLICE},
    {op_names::kGather, OpType::kGATHER},
    {op_names::kSqrt, OpType::kSQRT},
    {op_names::kErf, OpType::kERF},
    {op_names::kPow, OpType::kPOW},

    // Comparison and Logical
    {"Equal", OpType::kEQUAL},
    {"Where", OpType::kWHERE},
    {"Expand", OpType::kEXPAND},
    {"ConstantOfShape", OpType::kCONSTANT_OF_SHAPE},
};

// Static mapping table: OpType -> String
// For reverse lookup
const std::unordered_map<OpType, const char*> kOpTypeToStringMap = {
    // Convolution
    {OpType::kCONVOLUTION, op_names::kConv},
    {OpType::kCONV_TRANSPOSE, op_names::kConvTranspose},

    // Activation
    {OpType::kRELU, op_names::kRelu},
    {OpType::kSIGMOID, op_names::kSigmoid},
    {OpType::kTANH, op_names::kTanh},
    {OpType::kLEAKY_RELU, op_names::kLeakyRelu},
    {OpType::kPRELU, op_names::kPRelu},
    {OpType::kELU, op_names::kElu},
    {OpType::kGELU, op_names::kGelu},

    // Pooling
    {OpType::kMAX_POOL, op_names::kMaxPool},
    {OpType::kAVERAGE_POOL, op_names::kAveragePool},
    {OpType::kGLOBAL_AVERAGE_POOL, op_names::kGlobalAveragePool},
    {OpType::kGLOBAL_MAX_POOL, op_names::kGlobalMaxPool},

    // Normalization
    {OpType::kBATCH_NORM, op_names::kBatchNormalization},
    {OpType::kINSTANCE_NORM, op_names::kInstanceNormalization},
    {OpType::kLAYER_NORM, op_names::kLayerNormalization},
    {OpType::kLRN, op_names::kLRN},

    // Linear
    {OpType::kGEMM, op_names::kGemm},
    {OpType::kMATMUL, op_names::kMatMul},
    {OpType::kLINEAR, op_names::kLinear},

    // Shape
    {OpType::kRESHAPE, op_names::kReshape},
    {OpType::kFLATTEN, op_names::kFlatten},
    {OpType::kTRANSPOSE, op_names::kTranspose},
    {OpType::kSQUEEZE, op_names::kSqueeze},
    {OpType::kUNSQUEEZE, op_names::kUnsqueeze},
    {OpType::kCONCAT, op_names::kConcat},
    {OpType::kSPLIT, op_names::kSplit},
    {OpType::kSHAPE, op_names::kShape},

    // Element-wise
    {OpType::kADD, op_names::kAdd},
    {OpType::kSUB, op_names::kSub},
    {OpType::kMUL, op_names::kMul},
    {OpType::kDIV, op_names::kDiv},

    // Reduction
    {OpType::kREDUCE_SUM, op_names::kReduceSum},
    {OpType::kREDUCE_MEAN, op_names::kReduceMean},
    {OpType::kREDUCE_MAX, op_names::kReduceMax},
    {OpType::kREDUCE_MIN, op_names::kReduceMin},

    // Special
    {OpType::kSOFTMAX, op_names::kSoftmax},
    {OpType::kCAST, op_names::kCast},
    {OpType::kPADDING, op_names::kPad},
    {OpType::kSLICE, op_names::kSlice},
    {OpType::kGATHER, op_names::kGather},
    {OpType::kSQRT, op_names::kSqrt},
    {OpType::kERF, op_names::kErf},
    {OpType::kPOW, op_names::kPow},

    // Comparison and Logical
    {OpType::kEQUAL, "Equal"},
    {OpType::kWHERE, "Where"},
    {OpType::kEXPAND, "Expand"},
    {OpType::kCONSTANT_OF_SHAPE, "ConstantOfShape"},

    // Special cases
    {OpType::kCUSTOM, "Custom"},
    {OpType::kUNKNOWN, "Unknown"},
};

}  // anonymous namespace

OpType string_to_op_type(const std::string& op_name) {
    auto it = kStringToOpTypeMap.find(op_name);
    if (it != kStringToOpTypeMap.end()) {
        return it->second;
    }

    // Unknown operator -> treat as custom
    return OpType::kCUSTOM;
}

const char* op_type_to_string(OpType op_type) {
    auto it = kOpTypeToStringMap.find(op_type);
    if (it != kOpTypeToStringMap.end()) {
        return it->second;
    }

    return "Unknown";
}

}  // namespace core
}  // namespace mini_infer

// Include ActivationType definition for the conversion function
#include "mini_infer/operators/activation_type.h"

namespace mini_infer {
namespace core {

bool op_type_to_activation_type(OpType op_type, operators::ActivationType& act_type) {
    switch (op_type) {
        case OpType::kRELU:
            act_type = operators::ActivationType::RELU;
            return true;
        case OpType::kSIGMOID:
            act_type = operators::ActivationType::SIGMOID;
            return true;
        case OpType::kTANH:
            act_type = operators::ActivationType::TANH;
            return true;
        case OpType::kLEAKY_RELU:
            act_type = operators::ActivationType::LEAKY_RELU;
            return true;
        case OpType::kELU:
            act_type = operators::ActivationType::ELU;
            return true;
        case OpType::kPRELU:
            // PReLU requires parameters, treat as unsupported for simple fusion
            return false;
        default:
            return false;
    }
}

}  // namespace core
}  // namespace mini_infer
