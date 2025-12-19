#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "mini_infer/backends/device_context.h"
#include "mini_infer/core/op_type.h"
#include "mini_infer/core/tensor.h"
#include "mini_infer/core/types.h"

namespace mini_infer {
namespace kernels {

struct KernelContext {
    const std::vector<std::shared_ptr<core::Tensor>>* inputs{nullptr};
    std::vector<std::shared_ptr<core::Tensor>>* outputs{nullptr};
    std::shared_ptr<void> workspace;
    size_t workspace_size{0};
    const void* op_param{nullptr};
    backends::DeviceContext* device_context{nullptr};

    template <typename T>
    const T* param() const {
        return static_cast<const T*>(op_param);
    }
};

using KernelFunc = std::function<void(KernelContext*)>;

inline thread_local backends::DeviceContext* g_current_device_context = nullptr;

inline void set_current_device_context(backends::DeviceContext* context) {
    g_current_device_context = context;
}

inline backends::DeviceContext* get_current_device_context() {
    return g_current_device_context;
}

class KernelRegistry {
public:
    static KernelRegistry& instance() {
        static KernelRegistry registry;
        return registry;
    }

    void register_kernel(core::OpType op_type, core::DeviceType device_type, core::DataType dtype,
                         KernelFunc func) {
        registry_[Key{op_type, device_type, dtype}] = std::move(func);
    }

    KernelFunc find(core::OpType op_type, core::DeviceType device_type,
                    core::DataType dtype) const {
        auto it = registry_.find(Key{op_type, device_type, dtype});
        if (it == registry_.end()) {
            return nullptr;
        }
        return it->second;
    }

private:
    struct Key {
        core::OpType op_type;
        core::DeviceType device_type;
        core::DataType dtype;

        bool operator==(const Key& other) const {
            return op_type == other.op_type && device_type == other.device_type &&
                   dtype == other.dtype;
        }
    };

    struct KeyHash {
        size_t operator()(const Key& key) const {
            const auto op_hash = std::hash<int>{}(static_cast<int>(key.op_type));
            const auto dev_hash = std::hash<int>{}(static_cast<int>(key.device_type));
            const auto dtype_hash = std::hash<int>{}(static_cast<int>(key.dtype));
            size_t seed = op_hash;
            seed ^= dev_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= dtype_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    std::unordered_map<Key, KernelFunc, KeyHash> registry_;
};

}  // namespace kernels
}  // namespace mini_infer
