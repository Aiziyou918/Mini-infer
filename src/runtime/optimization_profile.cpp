#include "mini_infer/runtime/optimization_profile.h"
#include "mini_infer/utils/logger.h"
#include <sstream>
#include <algorithm>

namespace mini_infer {
namespace runtime {

// ============================================================================
// ShapeRange implementation
// ============================================================================

bool ShapeRange::is_valid() const {
    // All shapes must have same ndim
    if (min.ndim() != opt.ndim() || opt.ndim() != max.ndim()) {
        return false;
    }
    
    // For each dimension: min <= opt <= max (skip dynamic dimensions)
    for (size_t i = 0; i < min.ndim(); ++i) {
        int64_t min_dim = min[i];
        int64_t opt_dim = opt[i];
        int64_t max_dim = max[i];
        
        // Skip if any dimension is dynamic (-1)
        if (min_dim < 0 || opt_dim < 0 || max_dim < 0) {
            continue;
        }
        
        // Check ordering: min <= opt <= max
        if (!(min_dim <= opt_dim && opt_dim <= max_dim)) {
            return false;
        }
    }
    
    return true;
}

bool ShapeRange::contains(const core::Shape& shape) const {
    // Shape must have same ndim as range
    if (shape.ndim() != min.ndim()) {
        return false;
    }
    
    // Check each dimension is within [min, max]
    for (size_t i = 0; i < shape.ndim(); ++i) {
        int64_t dim = shape[i];
        int64_t min_dim = min[i];
        int64_t max_dim = max[i];
        
        // Skip dynamic dimensions in range definition
        if (min_dim < 0 || max_dim < 0) {
            continue;
        }
        
        // Check bounds
        if (dim < min_dim || dim > max_dim) {
            return false;
        }
    }
    
    return true;
}

std::string ShapeRange::to_string() const {
    std::stringstream ss;
    ss << "ShapeRange{";
    ss << "min=" << min.to_string() << ", ";
    ss << "opt=" << opt.to_string() << ", ";
    ss << "max=" << max.to_string();
    ss << "}";
    return ss.str();
}

// ============================================================================
// OptimizationProfile implementation
// ============================================================================

core::Status OptimizationProfile::set_shape_range(
    const std::string& input_name,
    const core::Shape& min,
    const core::Shape& opt,
    const core::Shape& max
) {
    // Create shape range
    ShapeRange range(min, opt, max);
    
    // Validate range
    if (!range.is_valid()) {
        MI_LOG_ERROR("[OptimizationProfile] Invalid shape range for input '" + 
                     input_name + "': " + range.to_string());
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Store range
    shape_ranges_[input_name] = range;
    
    MI_LOG_INFO("[OptimizationProfile] Set shape range for '" + input_name + "': " + 
                range.to_string());
    
    return core::Status::SUCCESS;
}

const ShapeRange* OptimizationProfile::get_shape_range(const std::string& input_name) const {
    auto it = shape_ranges_.find(input_name);
    if (it == shape_ranges_.end()) {
        return nullptr;
    }
    return &it->second;
}

bool OptimizationProfile::is_valid_for(const std::map<std::string, core::Shape>& shapes) const {
    // Check each input in the profile
    for (const auto& [name, range] : shape_ranges_) {
        // Find corresponding shape
        auto it = shapes.find(name);
        if (it == shapes.end()) {
            MI_LOG_WARNING("[OptimizationProfile] Input '" + name + 
                          "' not found in provided shapes");
            return false;
        }
        
        // Check if shape is within range
        if (!range.contains(it->second)) {
            MI_LOG_WARNING("[OptimizationProfile] Shape " + it->second.to_string() + 
                          " for input '" + name + "' is out of range " + range.to_string());
            return false;
        }
    }
    
    return true;
}

std::vector<std::string> OptimizationProfile::get_input_names() const {
    std::vector<std::string> names;
    names.reserve(shape_ranges_.size());
    
    for (const auto& [name, _] : shape_ranges_) {
        names.push_back(name);
    }
    
    return names;
}

std::map<std::string, core::Shape> OptimizationProfile::get_optimal_shapes() const {
    std::map<std::string, core::Shape> optimal_shapes;
    
    for (const auto& [name, range] : shape_ranges_) {
        optimal_shapes[name] = range.opt;
    }
    
    return optimal_shapes;
}

std::string OptimizationProfile::to_string() const {
    std::stringstream ss;
    ss << "OptimizationProfile{" << std::endl;
    
    for (const auto& [name, range] : shape_ranges_) {
        ss << "  " << name << ": " << range.to_string() << std::endl;
    }
    
    ss << "}";
    return ss.str();
}

} // namespace runtime
} // namespace mini_infer


