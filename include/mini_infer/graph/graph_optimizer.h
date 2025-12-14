#pragma once

#include "mini_infer/graph/graph.h"
#include "mini_infer/core/types.h"
#include <memory>
#include <vector>
#include <string>

namespace mini_infer {
namespace graph {

/**
 * @brief Graph Optimization Pass interface
 * 
 * Similar to TensorRT's IOptimizationPass.
 * Each pass performs a specific optimization on the graph.
 */
class OptimizationPass {
public:
    explicit OptimizationPass(const std::string& name) : name_(name) {}
    virtual ~OptimizationPass() = default;

    /**
     * @brief Apply optimization to graph
     * @param graph Graph to optimize
     * @return Status and number of modifications made
     */
    virtual core::Status apply(Graph* graph, int& num_modifications) = 0;

    /**
     * @brief Get pass name
     */
    const std::string& name() const { return name_; }

protected:
    std::string name_;
};

/**
 * @brief Optimization Pass Registry (TensorRT-style)
 * 
 * Global registry for optimization passes.
 * Passes can auto-register themselves using REGISTER_OPTIMIZATION_PASS macro.
 */
class OptimizationPassRegistry {
public:
    using PassCreator = std::shared_ptr<OptimizationPass>(*)();

    /**
     * @brief Get singleton instance
     */
    static OptimizationPassRegistry& instance();

    /**
     * @brief Register an optimization pass
     */
    void register_pass(const std::string& name, PassCreator creator, int priority = 100);

    /**
     * @brief Get all registered passes (sorted by priority)
     */
    std::vector<std::shared_ptr<OptimizationPass>> get_default_passes() const;

    /**
     * @brief Check if a pass is registered
     */
    bool has_pass(const std::string& name) const;

private:
    OptimizationPassRegistry() = default;
    
    struct PassInfo {
        std::string name;
        PassCreator creator;
        int priority; // Lower number = higher priority
        
        bool operator<(const PassInfo& other) const {
            return priority < other.priority;
        }
    };
    
    std::vector<PassInfo> passes_;
};

/**
 * @brief Graph Optimizer - Manages optimization passes
 * 
 * Similar to TensorRT's optimization pipeline.
 * Applies multiple optimization passes in sequence.
 */
class GraphOptimizer {
public:
    /**
     * @brief Create optimizer with default passes from registry
     */
    static GraphOptimizer create_default();

    GraphOptimizer() = default;
    ~GraphOptimizer() = default;

    /**
     * @brief Add an optimization pass
     * @param pass The optimization pass to add
     */
    void add_pass(std::shared_ptr<OptimizationPass> pass);

    /**
     * @brief Load all registered passes from registry
     */
    void load_default_passes();

    /**
     * @brief Optimize the graph
     * @param graph Graph to optimize
     * @return Status code
     */
    core::Status optimize(Graph* graph);

    /**
     * @brief Enable/disable verbose logging
     */
    void set_verbose(bool verbose) { verbose_ = verbose; }

    /**
     * @brief Get optimization statistics
     */
    struct Statistics {
        int total_passes = 0;
        int total_modifications = 0;
        std::vector<std::pair<std::string, int>> pass_results;
    };

    const Statistics& get_statistics() const { return stats_; }

private:
    std::vector<std::shared_ptr<OptimizationPass>> passes_;
    bool verbose_ = false;
    Statistics stats_;
};

/**
 * @brief Auto-registration helper for optimization passes
 */
#define REGISTER_OPTIMIZATION_PASS(PassClass, Priority)                           \
    namespace {                                                                    \
    std::shared_ptr<mini_infer::graph::OptimizationPass> create_##PassClass() {   \
        return std::make_shared<PassClass>();                                      \
    }                                                                              \
    struct PassClass##_Register {                                                  \
        PassClass##_Register() {                                                   \
            mini_infer::graph::OptimizationPassRegistry::instance()                \
                .register_pass(#PassClass, create_##PassClass, Priority);          \
        }                                                                          \
    };                                                                             \
    static PassClass##_Register g_##PassClass##_register;                          \
    }

} // namespace graph
} // namespace mini_infer
