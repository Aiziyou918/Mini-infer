#pragma once

#include "mini_infer/core/tensor.h"
#include "mini_infer/core/types.h"
#include <string>
#include <map>
#include <vector>

namespace mini_infer {
namespace runtime {

/**
 * @brief Shape range for optimization profile
 * 
 * Defines the valid range of input shapes for dynamic shape inference.
 * Following TensorRT's design pattern.
 */
struct ShapeRange {
    core::Shape min;   ///< Minimum shape (lower bound)
    core::Shape opt;   ///< Optimal shape (for optimization and default allocation)
    core::Shape max;   ///< Maximum shape (upper bound)
    
    ShapeRange() = default;
    
    /**
     * @brief Construct a shape range
     * 
     * @param min_ Minimum shape
     * @param opt_ Optimal shape (used for kernel selection)
     * @param max_ Maximum shape
     */
    ShapeRange(const core::Shape& min_, const core::Shape& opt_, const core::Shape& max_)
        : min(min_), opt(opt_), max(max_) {}
    
    /**
     * @brief Check if the shape range is valid
     * 
     * A valid range must satisfy:
     * - All shapes have the same number of dimensions
     * - For each dimension: min <= opt <= max
     * 
     * @return True if valid, false otherwise
     */
    bool is_valid() const;
    
    /**
     * @brief Check if a shape is within this range
     * 
     * @param shape Shape to check
     * @return True if shape is within [min, max], false otherwise
     */
    bool contains(const core::Shape& shape) const;
    
    /**
     * @brief Convert to string for debugging
     */
    std::string to_string() const;
};

/**
 * @brief Optimization Profile (TensorRT-style)
 * 
 * An optimization profile specifies the range of valid shapes for each
 * dynamic input tensor. The engine uses the optimal shape for:
 * - Memory allocation planning
 * - Kernel selection and optimization
 * - Build-time shape inference
 * 
 * Reference: TensorRT IOptimizationProfile
 * https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_optimization_profile.html
 * 
 * Example usage:
 * @code
 * auto profile = std::make_shared<OptimizationProfile>();
 * 
 * // Define shape range for input tensor
 * profile->set_shape_range("input",
 *     Shape({1, 3, 224, 224}),   // min: single image, low res
 *     Shape({4, 3, 384, 384}),   // opt: small batch, medium res
 *     Shape({8, 3, 512, 512})    // max: large batch, high res
 * );
 * 
 * // Use in engine config
 * EngineConfig config;
 * config.enable_dynamic_shapes = true;
 * config.optimization_profile = profile;
 * 
 * Engine engine(config);
 * engine.build(graph);  // Uses optimal shapes for optimization
 * 
 * // At runtime, any shape within range is valid
 * engine.forward(input_224);  // OK
 * engine.forward(input_512);  // OK
 * @endcode
 */
class OptimizationProfile {
public:
    OptimizationProfile() = default;
    ~OptimizationProfile() = default;
    
    /**
     * @brief Set shape range for an input tensor
     * 
     * @param input_name Name of the input tensor
     * @param min Minimum shape (lower bound)
     * @param opt Optimal shape (for optimization)
     * @param max Maximum shape (upper bound)
     * @return Status::SUCCESS if valid, error otherwise
     * 
     * The range must be valid:
     * - All shapes must have the same ndim
     * - For each dimension: min[i] <= opt[i] <= max[i]
     * - Dimensions can be -1 (dynamic) but should be consistent
     */
    core::Status set_shape_range(
        const std::string& input_name,
        const core::Shape& min,
        const core::Shape& opt,
        const core::Shape& max
    );
    
    /**
     * @brief Get shape range for an input tensor
     * 
     * @param input_name Name of the input tensor
     * @return Pointer to ShapeRange if found, nullptr otherwise
     */
    const ShapeRange* get_shape_range(const std::string& input_name) const;
    
    /**
     * @brief Check if a set of input shapes is valid for this profile
     * 
     * All input shapes must be within their respective ranges.
     * 
     * @param shapes Map of input name to actual shape
     * @return True if all shapes are valid, false otherwise
     */
    bool is_valid_for(const std::map<std::string, core::Shape>& shapes) const;
    
    /**
     * @brief Get all input names in this profile
     * 
     * @return Vector of input names
     */
    std::vector<std::string> get_input_names() const;
    
    /**
     * @brief Get the optimal shapes for all inputs
     * 
     * Used during Engine::build() for optimization.
     * 
     * @return Map of input name to optimal shape
     */
    std::map<std::string, core::Shape> get_optimal_shapes() const;

    /**
    * @brief Get the optimal shapes for all inputs
    *
    * Used during Engine::build() for optimization.
    *
    * @return Map of input name to optimal shape
    */
    std::map<std::string, core::Shape> get_max_shapes() const;
    
    /**
     * @brief Check if profile is empty
     */
    bool empty() const { return shape_ranges_.empty(); }
    
    /**
     * @brief Get number of inputs in profile
     */
    size_t size() const { return shape_ranges_.size(); }
    
    /**
     * @brief Clear all shape ranges
     */
    void clear() { shape_ranges_.clear(); }
    
    /**
     * @brief Convert to string for debugging
     */
    std::string to_string() const;
    
private:
    // Map of input name to shape range
    std::map<std::string, ShapeRange> shape_ranges_;
};

} // namespace runtime
} // namespace mini_infer


