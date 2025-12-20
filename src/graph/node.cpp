#include "mini_infer/graph/node.h"
#include "mini_infer/core/op_type.h"

namespace mini_infer {
namespace graph {

Node::Node(const std::string& name) : name_(name), cached_op_type_(core::OpType::kUNKNOWN) {}

void Node::set_operator(std::shared_ptr<operators::Operator> op) {
    op_ = op;

    // Cache OpType for fast access during graph optimization
    cached_op_type_ = op_ ? op_->type() : core::OpType::kUNKNOWN;
}

const char* Node::type_name() const {
    return op_ ? op_->name().c_str() : "Unknown";
}

void Node::add_input(const std::shared_ptr<Node>& input_node, int src_port, int dst_port) {
    if (input_node) {
        input_edges_.push_back(Edge{input_node, src_port, dst_port});
    }
}

void Node::add_output(const std::shared_ptr<Node>& output_node, int src_port, int dst_port) {
    if (output_node) {
        output_edges_.push_back(Edge{output_node, src_port, dst_port});
    }
}

}  // namespace graph
}  // namespace mini_infer
