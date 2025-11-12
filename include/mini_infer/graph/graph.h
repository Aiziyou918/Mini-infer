#pragma once

#include "mini_infer/graph/node.h"
#include "mini_infer/core/types.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mini_infer {
namespace graph {

/**
 * @brief Graph: manage nodes, connections and execution order.
 */
class Graph {
public:
    Graph() = default;
    ~Graph() = default;

    // Non-copyable, movable (避免误复制整图，提高语义清晰度)
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) noexcept = default;
    Graph& operator=(Graph&&) noexcept = default;

    /**
     * @brief Create or get a node with given name.
     * If a node with the same name already exists, returns the existing one.
     */
    std::shared_ptr<Node> create_node(const std::string& name);

    /**
     * @brief Get node by name.
     * @return nullptr if not found.
     */
    std::shared_ptr<Node> get_node(const std::string& name) const;

    /**
     * @brief Add a pre-constructed node into the graph.
     * If a node with the same name exists, it will be overwritten.
     */
    void add_node(const std::shared_ptr<Node>& node);

    /**
     * @brief Connect two nodes by name: src -> dst.
     * This will:
     *  - validate both nodes exist
     *  - reject self-loop (src == dst)
     *  - avoid duplicate edges (idempotent)
     */
    [[nodiscard]] core::Status connect(const std::string& src_name,
                                       const std::string& dst_name);

    /**
     * @brief Set the graph input node names.
     * Only stores names; existence will be checked in validate().
     */
    void set_inputs(const std::vector<std::string>& input_names);

    /**
     * @brief Set the graph output node names.
     * Only stores names; existence will be checked in validate().
     */
    void set_outputs(const std::vector<std::string>& output_names);

    /**
     * @brief Get input node names.
     */
    const std::vector<std::string>& inputs() const noexcept { return input_names_; }

    /**
     * @brief Get output node names.
     */
    const std::vector<std::string>& outputs() const noexcept { return output_names_; }

    /**
     * @brief Topological sort of the whole graph.
     *
     * Uses Kahn algorithm:
     *  - O(V + E)
     *  - If there's a cycle, returns ERROR_RUNTIME.
     */
    [[nodiscard]] core::Status
    topological_sort(std::vector<std::shared_ptr<Node>>& sorted_nodes) const;

    /**
     * @brief Validate inputs/outputs and obtain a topological order.
     *
     * Ensures declared graph inputs/outputs exist, then performs
     * topological_sort to guarantee the graph is acyclic while returning the
     * ordering.
     */
    [[nodiscard]] core::Status checked_topological_sort(
        std::vector<std::shared_ptr<Node>>& sorted_nodes) const;

    /**
     * @brief Graph-level optimization placeholder.
     * You can implement:
     *  - operator fusion
     *  - constant folding
     *  - dead node elimination, etc.
     */
    [[nodiscard]] core::Status optimize();

    /**
     * @brief Validate overall graph consistency.
     *
     * Checks:
     *  - all inputs/outputs exist
     *  - graph is acyclic (via topological_sort)
     */
    [[nodiscard]] core::Status validate() const;

    /**
     * @brief Access all nodes.
     */
    const std::unordered_map<std::string, std::shared_ptr<Node>>&
    nodes() const noexcept {
        return nodes_;
    }

    /**
     * @brief Helper: check if a name is marked as graph input.
     */
    bool is_input(const std::string& name) const;

    /**
     * @brief Helper: check if a name is marked as graph output.
     */
    bool is_output(const std::string& name) const;

private:
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

} // namespace graph
} // namespace mini_infer
