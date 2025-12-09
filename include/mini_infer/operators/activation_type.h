#pragma once

namespace mini_infer {
namespace operators {

/**
 * @brief Activation types (TensorRT-style)
 * 
 * Reference: TensorRT ActivationType enum
 * Defines activation functions that can be fused with other operators
 */
enum class ActivationType {
    NONE = 0,      // No activation
    RELU,          // ReLU: max(0, x)
    SIGMOID,       // Sigmoid: 1 / (1 + exp(-x))
    TANH,          // Tanh: tanh(x)
    LEAKY_RELU,    // LeakyReLU: x > 0 ? x : alpha * x
    ELU,           // ELU: x > 0 ? x : alpha * (exp(x) - 1)
    SELU,          // SELU: scale * (x > 0 ? x : alpha * (exp(x) - 1))
    SOFTSIGN,      // Softsign: x / (1 + |x|)
    SOFTPLUS,      // Softplus: log(exp(x) + 1)
    CLIP,          // Clip: min(max(x, alpha), beta)
    HARD_SIGMOID,  // HardSigmoid: max(0, min(1, alpha * x + beta))
    SCALED_TANH,   // ScaledTanh: alpha * tanh(beta * x)
    THRESHOLDED_RELU // ThresholdedReLU: x > alpha ? x : 0
};

/**
 * @brief Activation parameters
 * 
 * Used for parameterized activations (LeakyReLU, ELU, etc.)
 */
struct ActivationParam {
    ActivationType type = ActivationType::NONE;
    float alpha = 0.0f;  // Used by LeakyReLU, ELU, etc.
    float beta = 0.0f;   // Used by Clip, HardSigmoid, etc.
    
    ActivationParam() = default;
    
    ActivationParam(ActivationType t, float a = 0.0f, float b = 0.0f)
        : type(t), alpha(a), beta(b) {}
    
    // Check if activation is enabled
    bool is_enabled() const {
        return type != ActivationType::NONE;
    }
    
    // Get activation name
    const char* name() const {
        switch (type) {
            case ActivationType::NONE: return "None";
            case ActivationType::RELU: return "ReLU";
            case ActivationType::SIGMOID: return "Sigmoid";
            case ActivationType::TANH: return "Tanh";
            case ActivationType::LEAKY_RELU: return "LeakyReLU";
            case ActivationType::ELU: return "ELU";
            case ActivationType::SELU: return "SELU";
            case ActivationType::SOFTSIGN: return "Softsign";
            case ActivationType::SOFTPLUS: return "Softplus";
            case ActivationType::CLIP: return "Clip";
            case ActivationType::HARD_SIGMOID: return "HardSigmoid";
            case ActivationType::SCALED_TANH: return "ScaledTanh";
            case ActivationType::THRESHOLDED_RELU: return "ThresholdedReLU";
            default: return "Unknown";
        }
    }
};

/**
 * @brief Apply activation function inline
 * 
 * TensorRT-style inline activation for kernel fusion
 * 
 * @param value Input value
 * @param param Activation parameters
 * @return Activated value
 */
inline float apply_activation(float value, const ActivationParam& param) {
    switch (param.type) {
        case ActivationType::NONE:
            return value;
            
        case ActivationType::RELU:
            return value > 0.0f ? value : 0.0f;
            
        case ActivationType::SIGMOID:
            return 1.0f / (1.0f + std::exp(-value));
            
        case ActivationType::TANH:
            return std::tanh(value);
            
        case ActivationType::LEAKY_RELU:
            return value > 0.0f ? value : param.alpha * value;
            
        case ActivationType::ELU:
            return value > 0.0f ? value : param.alpha * (std::exp(value) - 1.0f);
            
        case ActivationType::SELU: {
            constexpr float scale = 1.0507009873554804934193349852946f;
            constexpr float alpha = 1.6732632423543772848170429916717f;
            return scale * (value > 0.0f ? value : alpha * (std::exp(value) - 1.0f));
        }
            
        case ActivationType::SOFTSIGN:
            return value / (1.0f + std::abs(value));
            
        case ActivationType::SOFTPLUS:
            return std::log(std::exp(value) + 1.0f);
            
        case ActivationType::CLIP:
            return std::min(std::max(value, param.alpha), param.beta);
            
        case ActivationType::HARD_SIGMOID:
            return std::max(0.0f, std::min(1.0f, param.alpha * value + param.beta));
            
        case ActivationType::SCALED_TANH:
            return param.alpha * std::tanh(param.beta * value);
            
        case ActivationType::THRESHOLDED_RELU:
            return value > param.alpha ? value : 0.0f;
            
        default:
            return value;
    }
}

} // namespace operators
} // namespace mini_infer
