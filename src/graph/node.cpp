#include "mini_infer/graph/node.h"

namespace mini_infer {
namespace graph {

Node::Node(const std::string& name) : name_(name), cached_op_type_(core::OpType::kUNKNOWN) {}

void Node::set_operator(std::shared_ptr<operators::Operator> op) {
    op_ = op;

    // Cache OpType for fast access during graph optimization
    if (op_) {
        cached_op_type_ = core::string_to_op_type(op_->name());
    } else {
        cached_op_type_ = core::OpType::kUNKNOWN;
    }
}

const char* Node::type_name() const {
    if (op_) {
        return op_->name().c_str();  // Convert std::string to const char*
    }
    return "Unknown";
}

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

}  // namespace graph
}  // namespace mini_infer
