#ifdef MINI_INFER_ONNX_ENABLED

#include "importers/internal/attribute_helper.h"
#include "onnx.pb.h"

namespace mini_infer {
namespace importers {

AttributeHelper::AttributeHelper(const onnx::NodeProto& node)
    : node_(node) {
}

const onnx::AttributeProto* AttributeHelper::find_attribute(const std::string& name) const {
    for (int i = 0; i < node_.attribute_size(); ++i) {
        const auto& attr = node_.attribute(i);
        if (attr.name() == name) {
            return &attr;
        }
    }
    return nullptr;
}

int64_t AttributeHelper::get_int(const std::string& name, int64_t default_value) const {
    const auto* attr = find_attribute(name);
    if (attr && attr->has_i()) {
        return attr->i();
    }
    return default_value;
}

float AttributeHelper::get_float(const std::string& name, float default_value) const {
    const auto* attr = find_attribute(name);
    if (attr && attr->has_f()) {
        return attr->f();
    }
    return default_value;
}

std::string AttributeHelper::get_string(const std::string& name, const std::string& default_value) const {
    const auto* attr = find_attribute(name);
    if (attr && attr->has_s()) {
        return attr->s();
    }
    return default_value;
}

std::vector<int64_t> AttributeHelper::get_ints(const std::string& name) const {
    const auto* attr = find_attribute(name);
    if (attr && attr->ints_size() > 0) {
        return std::vector<int64_t>(attr->ints().begin(), attr->ints().end());
    }
    return {};
}

std::vector<float> AttributeHelper::get_floats(const std::string& name) const {
    const auto* attr = find_attribute(name);
    if (attr && attr->floats_size() > 0) {
        return std::vector<float>(attr->floats().begin(), attr->floats().end());
    }
    return {};
}

std::vector<std::string> AttributeHelper::get_strings(const std::string& name) const {
    const auto* attr = find_attribute(name);
    if (attr && attr->strings_size() > 0) {
        return std::vector<std::string>(attr->strings().begin(), attr->strings().end());
    }
    return {};
}

bool AttributeHelper::has_attribute(const std::string& name) const {
    return find_attribute(name) != nullptr;
}

std::vector<std::string> AttributeHelper::get_attribute_names() const {
    std::vector<std::string> names;
    names.reserve(node_.attribute_size());
    for (int i = 0; i < node_.attribute_size(); ++i) {
        names.push_back(node_.attribute(i).name());
    }
    return names;
}

} // namespace importers
} // namespace mini_infer

#endif // MINI_INFER_ONNX_ENABLED
