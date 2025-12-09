#include "mini_infer/graph/graph_optimizer.h"
#include "mini_infer/utils/logger.h"

namespace mini_infer {
namespace graph {

void GraphOptimizer::add_pass(std::shared_ptr<OptimizationPass> pass) {
    if (pass) {
        passes_.push_back(pass);
    }
}

core::Status GraphOptimizer::optimize(Graph* graph) {
    if (!graph) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Reset statistics
    stats_ = Statistics();
    stats_.total_passes = static_cast<int>(passes_.size());

    if (verbose_) {
        MI_LOG_INFO("[GraphOptimizer] Starting optimization with " + 
                    std::to_string(stats_.total_passes) + " passes");
    }

    // Apply each optimization pass
    for (auto& pass : passes_) {
        int num_modifications = 0;
        
        if (verbose_) {
            MI_LOG_INFO("[GraphOptimizer] Applying pass: " + pass->name());
        }

        auto status = pass->apply(graph, num_modifications);
        if (status != core::Status::SUCCESS) {
            MI_LOG_ERROR("[GraphOptimizer] Pass failed: " + pass->name());
            return status;
        }

        stats_.total_modifications += num_modifications;
        stats_.pass_results.push_back({pass->name(), num_modifications});

        if (verbose_) {
            MI_LOG_INFO("[GraphOptimizer] Pass completed: " + pass->name() + 
                        ", modifications: " + std::to_string(num_modifications));
        }
    }

    if (verbose_) {
        MI_LOG_INFO("[GraphOptimizer] Optimization completed. Total modifications: " + 
                    std::to_string(stats_.total_modifications));
    }

    return core::Status::SUCCESS;
}

} // namespace graph
} // namespace mini_infer
