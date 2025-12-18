#include "mini_infer/graph/graph_optimizer.h"
#include "mini_infer/utils/logger.h"
#include <algorithm>

namespace mini_infer {
namespace graph {

// ============================================================================
// OptimizationPassRegistry Implementation
// ============================================================================

OptimizationPassRegistry& OptimizationPassRegistry::instance() {
    static OptimizationPassRegistry registry;
    return registry;
}

void OptimizationPassRegistry::register_pass(const std::string& name, 
                                             PassCreator creator, 
                                             int priority) {
    PassInfo info;
    info.name = name;
    info.creator = creator;
    info.priority = priority;
    passes_.push_back(info);
    
    // Keep sorted by priority
    std::sort(passes_.begin(), passes_.end());
}

std::vector<std::shared_ptr<OptimizationPass>> 
OptimizationPassRegistry::get_default_passes() const {
    std::vector<std::shared_ptr<OptimizationPass>> result;
    result.reserve(passes_.size());
    
    for (const auto& info : passes_) {
        if (info.creator) {
            result.push_back(info.creator());
        }
    }
    
    return result;
}

bool OptimizationPassRegistry::has_pass(const std::string& name) const {
    for (const auto& info : passes_) {
        if (info.name == name) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// GraphOptimizer Implementation
// ============================================================================

GraphOptimizer GraphOptimizer:: create_default() {
    GraphOptimizer optimizer;
    optimizer.load_default_passes();
    return optimizer;
}

void GraphOptimizer::add_pass(std::shared_ptr<OptimizationPass> pass) {
    if (pass) {
        passes_.push_back(pass);
    }
}

void GraphOptimizer::load_default_passes() {
    auto default_passes = OptimizationPassRegistry::instance().get_default_passes();
    for (auto& pass : default_passes) {
        add_pass(pass);
    }
    
    if (verbose_) {
        MI_LOG_INFO("[GraphOptimizer] Loaded " + std::to_string(default_passes.size()) + 
                    " default optimization passes");
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
