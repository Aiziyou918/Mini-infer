#pragma once

#include "mini_infer/kernels/kernel_types.h"
#include <functional>
#include <vector>
#include <memory>
#include <algorithm>

namespace mini_infer {
namespace kernels {

/**
 * @brief Kernel Base - TensorRT-style kernel management
 * 
 * Provides a registry-based kernel dispatch system similar to TensorRT's plugin architecture.
 * This avoids virtual function overhead while maintaining flexibility.
 */

/**
 * @brief Kernel capability checker
 */
using KernelCapabilityChecker = std::function<bool()>;

/**
 * @brief Kernel registry entry
 * 
 * Each kernel implementation registers itself with backend type and capability checker.
 */
template<typename FuncType>
struct KernelEntry {
    KernelBackend backend;
    FuncType func;
    KernelCapabilityChecker is_supported;
    int priority;  // Higher priority = preferred
    
    KernelEntry(KernelBackend b, FuncType f, 
                KernelCapabilityChecker checker, int prio = 0)
        : backend(b), func(f), is_supported(checker), priority(prio) {}
};

/**
 * @brief Kernel registry base class
 * 
 * Template pattern for kernel registration and dispatch.
 * Similar to TensorRT's IPluginRegistry but for kernels.
 */
template<typename FuncType>
class KernelRegistryBase {
public:
    using Entry = KernelEntry<FuncType>;
    
    /**
     * @brief Register a kernel implementation
     */
    void register_kernel(KernelBackend backend, 
                        FuncType func,
                        KernelCapabilityChecker checker,
                        int priority = 0) {
        entries_.emplace_back(backend, func, checker, priority);
        
        // Sort by priority (descending)
        std::sort(entries_.begin(), entries_.end(),
                 [](const Entry& a, const Entry& b) {
                     return a.priority > b.priority;
                 });
    }
    
    /**
     * @brief Get best available kernel
     */
    FuncType get_best_kernel() const {
        for (const auto& entry : entries_) {
            if (entry.is_supported()) {
                return entry.func;
            }
        }
        return nullptr;
    }
    
    /**
     * @brief Get kernel for specific backend
     */
    FuncType get_kernel(KernelBackend backend) const {
        for (const auto& entry : entries_) {
            if (entry.backend == backend && entry.is_supported()) {
                return entry.func;
            }
        }
        return nullptr;
    }
    
    /**
     * @brief Check if backend is supported
     */
    bool is_backend_available(KernelBackend backend) const {
        for (const auto& entry : entries_) {
            if (entry.backend == backend && entry.is_supported()) {
                return true;
            }
        }
        return false;
    }
    
protected:
    std::vector<Entry> entries_;
};

/**
 * @brief Auto-registration helper
 * 
 * Usage:
 * namespace cpu {
 *     void gemm_impl(...) { ... }
 *     static auto _ = AutoRegister<GEMMRegistry>(
 *         KernelBackend::CPU, gemm_impl, []() { return true; }, 100);
 * }
 */
template<typename Registry, typename FuncType>
struct AutoRegister {
    AutoRegister(KernelBackend backend, 
                FuncType func,
                KernelCapabilityChecker checker,
                int priority = 0) {
        Registry::instance().register_kernel(backend, func, checker, priority);
    }
};

} // namespace kernels
} // namespace mini_infer
