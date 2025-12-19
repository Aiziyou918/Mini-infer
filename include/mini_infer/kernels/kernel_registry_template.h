#pragma once

#include "mini_infer/kernels/kernel_base.h"
#include <type_traits>

namespace mini_infer {
namespace kernels {

/**
 * @brief Generic Kernel Registry (Modern C++ approach)
 * 
 * Type-safe registry using template specialization.
 * Eliminates code duplication without macros.
 * 
 * Usage:
 * using GEMMRegistry_NN = KernelRegistryTemplate<GEMMFunc_NN>;
 * GEMMRegistry_NN<float>::instance().register_kernel(...);
 */
template<template<typename> class FuncType>
class KernelRegistryTemplate {
public:
    template<typename T>
    class ForType : public KernelRegistryBase<FuncType<T>> {
    public:
        static ForType& instance() {
            static ForType reg;
            return reg;
        }
        
    private:
        ForType() = default;
        ForType(const ForType&) = delete;
        ForType& operator=(const ForType&) = delete;
    };
};

/**
 * @brief Type alias for cleaner syntax
 * 
 * Instead of:
 *   KernelRegistryTemplate<GEMMFunc_NN>::ForType<float>::instance()
 * 
 * Use:
 *   GEMMRegistry_NN<float>::instance()
 */
template<typename T, template<typename> class FuncType>
using KernelRegistryFor = typename KernelRegistryTemplate<FuncType>::template ForType<T>;

// Example: Define registry alias
#define DEFINE_REGISTRY_ALIAS(Name, FuncType) \
    template<typename T> \
    using Name = KernelRegistryFor<T, FuncType>

/**
 * @brief Macro to define backend availability checker
 * 
 * Usage:
 * DEFINE_BACKEND_CHECKER(is_backend_available_nn, GEMMRegistry_NN)
 * 
 * Expands to:
 * template<typename T>
 * static bool is_backend_available_nn(KernelBackend backend) {
 *     return GEMMRegistry_NN<T>::instance().is_backend_available(backend);
 * }
 */
#define DEFINE_BACKEND_CHECKER(FuncName, RegistryType) \
    template<typename T> \
    static bool FuncName(KernelBackend backend) { \
        return RegistryType<T>::instance().is_backend_available(backend); \
    }

/**
 * @brief Macro to define best backend getter
 * 
 * Usage:
 * DEFINE_BEST_BACKEND_GETTER(get_best_backend_nn, GEMMRegistry_NN)
 */
#define DEFINE_BEST_BACKEND_GETTER(FuncName, RegistryType) \
    template<typename T> \
    static KernelBackend FuncName() { \
        return RegistryType<T>::instance().get_best_backend(); \
    }

} // namespace kernels
} // namespace mini_infer
