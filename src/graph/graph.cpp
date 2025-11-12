#include "mini_infer/graph/graph.h"

#include <queue>
#include <unordered_set>

namespace mini_infer {
namespace graph {

std::shared_ptr<Node> Graph::create_node(const std::string& name) {
    if (name.empty()) {
        return nullptr;
    }

    auto it = nodes_.find(name);
    if (it != nodes_.end()) {
        // Reuse existing node to avoid duplicate definition
        return it->second;
    }

    auto node = std::make_shared<Node>(name);
    nodes_.emplace(name, node);
    return node;
}

std::shared_ptr<Node> Graph::get_node(const std::string& name) const {
    auto it = nodes_.find(name);
    if (it == nodes_.end()) {
        return nullptr;
    }
    return it->second;
}

void Graph::add_node(const std::shared_ptr<Node>& node) {
    if (!node) {
        return;
    }
    const std::string& name = node->name();
    if (name.empty()) {
        return;
    }
    nodes_[name] = node;
}

core::Status Graph::connect(const std::string& src_name,
                            const std::string& dst_name) {
    if (src_name.empty() || dst_name.empty()) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    if (src_name == dst_name) {
        // Generally not allowed to have self-loops; if you need RNN-style, you can open it here and modify the topological logic
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto src = get_node(src_name);
    auto dst = get_node(dst_name);
    if (!src || !dst) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Avoid duplicate edges (idempotent)
    for (const auto& out : src->outputs()) {
        if (out && out->name() == dst_name) {
            return core::Status::SUCCESS;
        }
    }

    src->add_output(dst);
    dst->add_input(src);

    return core::Status::SUCCESS;
}

void Graph::set_inputs(const std::vector<std::string>& input_names) {
    input_names_ = input_names;
}

void Graph::set_outputs(const std::vector<std::string>& output_names) {
    output_names_ = output_names;
}

bool Graph::is_input(const std::string& name) const {
    for (const auto& n : input_names_) {
        if (n == name) {
            return true;
        }
    }
    return false;
}

bool Graph::is_output(const std::string& name) const {
    for (const auto& n : output_names_) {
        if (n == name) {
            return true;
        }
    }
    return false;
}

core::Status Graph::topological_sort(std::vector<std::shared_ptr<Node>>& sorted_nodes) const {
    sorted_nodes.clear();

    const size_t num_nodes = nodes_.size();
    if (num_nodes == 0) {
        return core::Status::SUCCESS;
    }

    // 1) Initialize the in-degree table
    std::unordered_map<std::string, int> in_degree;
    in_degree.reserve(num_nodes);
    for (const auto& kv : nodes_) {
        in_degree.emplace(kv.first, 0);
    }

    // 2) Count the in-degree: build directed edges according to outputs
    for (const auto& kv : nodes_) {
        const auto& node = kv.second;
        if (!node) continue;

        for (const auto& out : node->outputs()) {
            if (!out) {
                continue;
            }
            auto it = in_degree.find(out->name());
            if (it != in_degree.end()) {
                ++(it->second);
            }
            // If out is not in in_degree, it means the graph structure is inconsistent, generally it is a graph construction error
            // Here we do not report an error directly, but let validate()/caller check it
        }
    }

    // 3) Put the nodes with in-degree 0 into the queue
    std::queue<std::shared_ptr<Node>> q;
    for (const auto& kv : nodes_) {
        const auto& name = kv.first;
        const auto& node = kv.second;
        if (!node) continue;

        auto it = in_degree.find(name);
        if (it != in_degree.end() && it->second == 0) {
            q.push(node);
        }
    }

    // When the graph has a cycle, here will not cover all nodes
    sorted_nodes.reserve(num_nodes);

    // 4) Kahn algorithm
    while (!q.empty()) {
        auto node = q.front();
        q.pop();

        sorted_nodes.push_back(node);

        for (const auto& out : node->outputs()) {
            if (!out) continue;
            auto it = in_degree.find(out->name());
            if (it == in_degree.end()) {
                continue; // Nodes not in nodes_ are ignored; handled by validate()
            }

            --(it->second);
            if (it->second == 0) {
                q.push(out);
            }
        }
    }

    // 5) If not all nodes are covered, it means there is a cycle or illegal pointer
    if (sorted_nodes.size() != num_nodes) {
        return core::Status::ERROR_RUNTIME;
    }

    return core::Status::SUCCESS;
}

core::Status Graph::checked_topological_sort(
    std::vector<std::shared_ptr<Node>>& sorted_nodes) const {
    // 1) Check if the input/output names exist
    for (const auto& name : input_names_) {
        if (nodes_.find(name) == nodes_.end()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }

    for (const auto& name : output_names_) {
        if (nodes_.find(name) == nodes_.end()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }

    // 2) Perform topological sorting to ensure DAG property and obtain order
    return topological_sort(sorted_nodes);
}

core::Status Graph::optimize() {
    // TODO: Implement graph optimization (operator fusion, constant folding, etc.)
    // - constant folding
    // - dead node elimination (not reachable from inputs->outputs)
    // - operator fusion (Conv+BN, etc.)
    //
    // Currently return SUCCESS, indicating that the existing process is not broken.
    return core::Status::SUCCESS;
}

core::Status Graph::validate() const {
    std::vector<std::shared_ptr<Node>> topo;
    return checked_topological_sort(topo);
}

} // namespace graph
} // namespace mini_infer
