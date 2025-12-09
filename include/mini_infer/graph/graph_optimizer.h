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
 * @brief Graph Optimizer - Manages optimization passes
 * 
 * Similar to TensorRT's optimization pipeline.
 * Applies multiple optimization passes in sequence.
 */
class GraphOptimizer {
public:
    GraphOptimizer() = default;
    ~GraphOptimizer() = default;

    /**
     * @brief Add an optimization pass
     * @param pass The optimization pass to add
     */
    void add_pass(std::shared_ptr<OptimizationPass> pass);

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

} // namespace graph
} // namespace mini_infer
