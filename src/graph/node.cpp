#include "mini_infer/graph/node.h"

namespace mini_infer {
namespace graph {

Node::Node(const std::string& name) : name_(name) {}

void Node::add_input(std::shared_ptr<Node> input_node) {
    if (input_node) {
        input_nodes_.push_back(input_node);
    }
}

void Node::add_output(std::shared_ptr<Node> output_node) {
    if (output_node) {
        output_nodes_.push_back(output_node);
    }
}

} // namespace graph
} // namespace mini_infer

