#include "mini_infer/runtime/engine.h"

namespace mini_infer {
namespace runtime {

Engine::Engine(const EngineConfig& config) : plan_(std::make_shared<InferencePlan>(config)) {}

core::Status Engine::build(std::shared_ptr<graph::Graph> graph) {
    return plan_->build(std::move(graph));
}

std::shared_ptr<ExecutionContext> Engine::create_context() const {
    return plan_->create_execution_context();
}

std::vector<std::string> Engine::get_input_names() const {
    return plan_->get_input_names();
}

std::vector<std::string> Engine::get_output_names() const {
    return plan_->get_output_names();
}

std::string Engine::get_profiling_info() const {
    return "Profiling not implemented yet";
}

}  // namespace runtime
}  // namespace mini_infer
