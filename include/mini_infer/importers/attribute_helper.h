#pragma once

#include <string>
#include <vector>
#include <cstdint>

// Forward declarations
namespace onnx {
    class NodeProto;
    class AttributeProto;
}

namespace mini_infer {
namespace importers {

/**
 * @brief Attribute Helper - Utilities for parsing ONNX node attributes
 * 
 * Similar to TensorRT's attribute parsing utilities.
 * Provides type-safe attribute access with default values.
 */
class AttributeHelper {
public:
    explicit AttributeHelper(const onnx::NodeProto& node);
    ~AttributeHelper() = default;

    // Scalar attributes
    /**
     * @brief Get integer attribute
     * @param name Attribute name
     * @param default_value Default value if attribute not found
     * @return Attribute value or default
     */
    int64_t get_int(const std::string& name, int64_t default_value = 0) const;

    /**
     * @brief Get float attribute
     * @param name Attribute name
     * @param default_value Default value if attribute not found
     * @return Attribute value or default
     */
    float get_float(const std::string& name, float default_value = 0.0f) const;

    /**
     * @brief Get string attribute
     * @param name Attribute name
     * @param default_value Default value if attribute not found
     * @return Attribute value or default
     */
    std::string get_string(const std::string& name, const std::string& default_value = "") const;

    // Array attributes
    /**
     * @brief Get integer array attribute
     * @param name Attribute name
     * @return Vector of integers
     */
    std::vector<int64_t> get_ints(const std::string& name) const;

    /**
     * @brief Get float array attribute
     * @param name Attribute name
     * @return Vector of floats
     */
    std::vector<float> get_floats(const std::string& name) const;

    /**
     * @brief Get string array attribute
     * @param name Attribute name
     * @return Vector of strings
     */
    std::vector<std::string> get_strings(const std::string& name) const;

    // Existence check
    /**
     * @brief Check if attribute exists
     * @param name Attribute name
     * @return true if attribute exists
     */
    bool has_attribute(const std::string& name) const;

    /**
     * @brief Get all attribute names
     * @return Vector of attribute names
     */
    std::vector<std::string> get_attribute_names() const;

private:
    const onnx::NodeProto& node_;

    const onnx::AttributeProto* find_attribute(const std::string& name) const;
};

/**
 * @brief Helper function to convert ONNX int array to std::vector<int>
 */
inline std::vector<int> to_int_vector(const std::vector<int64_t>& int64_vec) {
    return std::vector<int>(int64_vec.begin(), int64_vec.end());
}

/**
 * @brief Helper function to convert ONNX int array to size_t vector
 */
inline std::vector<size_t> to_size_t_vector(const std::vector<int64_t>& int64_vec) {
    return std::vector<size_t>(int64_vec.begin(), int64_vec.end());
}

} // namespace importers
} // namespace mini_infer
