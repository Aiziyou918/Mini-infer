#include "mini_infer/operators/operator.h"
#include <unordered_map>

namespace mini_infer {
namespace operators {

std::unordered_map<std::string, OperatorFactory::CreateFunc>& 
OperatorFactory::get_registry() {
    static std::unordered_map<std::string, CreateFunc> registry;
    return registry;
}

void OperatorFactory::register_operator(const std::string& op_type, CreateFunc func) {
    get_registry()[op_type] = func;
}

std::shared_ptr<Operator> OperatorFactory::create_operator(const std::string& op_type) {
    auto& registry = get_registry();
    auto it = registry.find(op_type);
    if (it != registry.end()) {
        return it->second();
    }
    return nullptr;
}

} // namespace operators
} // namespace mini_infer

