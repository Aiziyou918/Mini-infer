#pragma once

#include <memory>
#include <string>

#include "mini_infer/runtime/execution_context.h"
#include "mini_infer/runtime/inference_plan.h"

namespace mini_infer {
namespace runtime {

/**
 * @brief Legacy Engine wrapper
 *
 * Delegates build-time work to InferencePlan and creates per-request
 * ExecutionContext instances.
 */
class Engine {
   public:
    explicit Engine(const EngineConfig& config);
    ~Engine() = default;

    core::Status build(std::shared_ptr<graph::Graph> graph);

    std::shared_ptr<InferencePlan> plan() const {
        return plan_;
    }

    std::shared_ptr<ExecutionContext> create_context() const;

    std::vector<std::string> get_input_names() const;
    std::vector<std::string> get_output_names() const;
    std::string get_profiling_info() const;

    const MemoryPlan& get_memory_plan() const {
        return plan_->get_memory_plan();
    }

    const graph::GraphOptimizer::Statistics& get_optimization_stats() const {
        return plan_->get_optimization_stats();
    }

   private:
    std::shared_ptr<InferencePlan> plan_;
};

}  // namespace runtime
}  // namespace mini_infer
