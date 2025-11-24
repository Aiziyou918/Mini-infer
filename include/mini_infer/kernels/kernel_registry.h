#pragma once

namespace mini_infer {
namespace kernels {

/**
 * @brief Kernel Registry Initializer
 * 
 * Force initialization of all kernel registries.
 * Call this at program startup to ensure all kernels are registered.
 * 
 * This is necessary because static initialization in static libraries
 * may be optimized away by the linker.
 */
class KernelRegistryInitializer {
public:
    /**
     * @brief Initialize all kernel registries
     * 
     * This function forces the linker to include kernel registration code
     * by explicitly referencing the registration functions.
     * 
     * Should be called once at program startup (e.g., in main() or static init).
     */
    static void initialize();
    
private:
    static bool initialized_;
};

} // namespace kernels
} // namespace mini_infer
