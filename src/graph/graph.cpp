#include "mini_infer/graph/graph.h"

#include <algorithm>
#include <queue>

#include "mini_infer/graph/graph_optimizer.h"

namespace mini_infer {
namespace graph {

std::shared_ptr<Node> Graph::create_node(const std::string& name) {
    if (name.empty()) {
        return nullptr;
    }

    auto it = name_to_id_.find(name);
    if (it != name_to_id_.end()) {
        // Reuse existing node to avoid duplicate definition
        return get_node(it->second);
    }

    auto node = std::make_shared<Node>(name);
    node->set_id(nodes_.size());
    nodes_.push_back(node);
    name_to_id_[name] = node->id();
    return node;
}

std::shared_ptr<Node> Graph::get_node(const std::string& name) const {
    auto it = name_to_id_.find(name);
    if (it == name_to_id_.end()) {
        return nullptr;
    }
    return get_node(it->second);
}

std::shared_ptr<Node> Graph::get_node(size_t id) const {
    if (id >= nodes_.size()) {
        return nullptr;
    }
    return nodes_[id];
}

void Graph::add_node(const std::shared_ptr<Node>& node) {
    if (!node) {
        return;
    }
    const std::string& name = node->name();
    if (name.empty()) {
        return;
    }
    auto it = name_to_id_.find(name);
    if (it != name_to_id_.end()) {
        const size_t id = it->second;
        node->set_id(id);
        nodes_[id] = node;
        return;
    }
    node->set_id(nodes_.size());
    nodes_.push_back(node);
    name_to_id_[name] = node->id();
}

void Graph::remove_node(const std::string& name) {
    auto it = name_to_id_.find(name);
    if (it == name_to_id_.end()) {
        return;
    }
    const size_t target_id = it->second;
    auto target = get_node(target_id);
    // Remove all edges pointing to the target node
    for (auto& node : nodes_) {
        if (!node || node == target) {
            continue;
        }
        // Remove all edges pointing to the target node
        auto& outs = node->mutable_outputs();
        outs.erase(std::remove_if(
                       outs.begin(), outs.end(),
                       [&](const Node::Edge& e) { return e.node && e.node->id() == target_id; }),
                   outs.end());
        // Remove all edges pointing to the target node
        auto& ins = node->mutable_inputs();
        ins.erase(std::remove_if(
                      ins.begin(), ins.end(),
                      [&](const Node::Edge& e) { return e.node && e.node->id() == target_id; }),
                  ins.end());
    }
    nodes_[target_id] = nullptr;
    name_to_id_.erase(it);
}

core::Status Graph::connect(size_t src_id, size_t dst_id, int src_port, int dst_port) {
    if (src_port < 0 || dst_port < 0) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    if (src_id == dst_id) {
        // Generally not allowed to have self-loops; if you need RNN-style, you can open it here and
        // modify the topological logic
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto src = get_node(src_id);
    auto dst = get_node(dst_id);
    if (!src || !dst) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Avoid duplicate edges (idempotent)
    for (const auto& out : src->outputs()) {
        if (out.node && out.node->id() == dst_id && out.src_port == src_port &&
            out.dst_port == dst_port) {
            return core::Status::SUCCESS;
        }
    }

    src->add_output(dst, src_port, dst_port);
    dst->add_input(src, src_port, dst_port);

    return core::Status::SUCCESS;
}

core::Status Graph::connect(const std::string& src_name, const std::string& dst_name, int src_port,
                            int dst_port) {
    if (src_name.empty() || dst_name.empty()) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    auto src = name_to_id_.find(src_name);
    auto dst = name_to_id_.find(dst_name);
    if (src == name_to_id_.end() || dst == name_to_id_.end()) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    return connect(src->second, dst->second, src_port, dst_port);
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

    const size_t capacity = nodes_.size();
    if (capacity == 0) {
        return core::Status::SUCCESS;
    }

    // 1) Initialize the in-degree table
    std::vector<int> in_degree(capacity, 0);
    std::vector<char> active(capacity, 0);
    size_t active_count = 0;
    for (const auto& node : nodes_) {
        if (!node) {
            continue;
        }
        active[node->id()] = 1;
        active_count++;
    }

    // 2) Count the in-degree: build directed edges according to outputs
    for (const auto& node : nodes_) {
        if (!node)
            continue;

        for (const auto& out : node->outputs()) {
            if (!out.node) {
                continue;
            }
            const size_t out_id = out.node->id();
            if (out_id < in_degree.size() && active[out_id]) {
                ++in_degree[out_id];
            }
            // If out is not in in_degree, it means the graph structure is inconsistent, generally
            // it is a graph construction error Here we do not report an error directly, but let
            // validate()/caller check it
        }
    }

    // 3) Put the nodes with in-degree 0 into the queue
    std::queue<std::shared_ptr<Node>> q;
    for (const auto& node : nodes_) {
        if (!node)
            continue;

        const size_t node_id = node->id();
        if (node_id < in_degree.size() && in_degree[node_id] == 0) {
            q.push(node);
        }
    }

    // When the graph has a cycle, here will not cover all nodes
    sorted_nodes.reserve(active_count);

    // 4) Kahn algorithm
    while (!q.empty()) {
        auto node = q.front();
        q.pop();

        sorted_nodes.push_back(node);

        for (const auto& out : node->outputs()) {
            if (!out.node)
                continue;
            const size_t out_id = out.node->id();
            if (out_id >= in_degree.size() || !active[out_id]) {
                continue;
            }

            --in_degree[out_id];
            if (in_degree[out_id] == 0) {
                q.push(out.node);
            }
        }
    }

    // 5) If not all nodes are covered, it means there is a cycle or illegal pointer
    if (sorted_nodes.size() != active_count) {
        return core::Status::ERROR_RUNTIME;
    }

    return core::Status::SUCCESS;
}

core::Status Graph::checked_topological_sort(
    std::vector<std::shared_ptr<Node>>& sorted_nodes) const {
    // 1) Check if the input/output names exist
    for (const auto& name : input_names_) {
        if (name_to_id_.find(name) == name_to_id_.end()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }

    for (const auto& name : output_names_) {
        if (name_to_id_.find(name) == name_to_id_.end()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }
    // 2) Perform topological sorting to ensure DAG property and obtain order
    return topological_sort(sorted_nodes);
}

core::Status Graph::optimize() {
    // Invoke the GraphOptimizer pipeline
    // This will load and execute all registered optimization passes (e.g. constant folding, fusion)
    auto optimizer = GraphOptimizer::create_default();

    // You can enable verbose logging if needed
    optimizer.set_verbose(true);

    return optimizer.optimize(this);
}

core::Status Graph::validate() const {
    std::vector<std::shared_ptr<Node>> topo;
    return checked_topological_sort(topo);
}

}  // namespace graph
}  // namespace mini_infer
