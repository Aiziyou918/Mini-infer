#include "mini_infer/graph/graph.h"
#include "mini_infer/graph/node.h"
#include <iostream>
#include <cassert>

using namespace mini_infer;

void test_graph_creation() {
    std::cout << "Testing graph creation..." << std::endl;
    
    graph::Graph g;
    
    auto node1 = g.create_node("input");
    auto node2 = g.create_node("conv");
    auto node3 = g.create_node("output");
    
    assert(node1 != nullptr);
    assert(node2 != nullptr);
    assert(node3 != nullptr);
    
    std::cout << "✓ Graph creation test passed" << std::endl;
}

void test_graph_connection() {
    std::cout << "Testing graph connection..." << std::endl;
    
    graph::Graph g;
    
    auto node1 = g.create_node("input");
    auto node2 = g.create_node("conv");
    auto node3 = g.create_node("output");
    
    auto status = g.connect("input", "conv");
    assert(status == core::Status::SUCCESS);
    
    status = g.connect("conv", "output");
    assert(status == core::Status::SUCCESS);
    
    assert(node1->outputs().size() == 1);
    assert(node2->inputs().size() == 1);
    assert(node2->outputs().size() == 1);
    assert(node3->inputs().size() == 1);
    
    std::cout << "✓ Graph connection test passed" << std::endl;
}

void test_topological_sort() {
    std::cout << "Testing topological sort..." << std::endl;
    
    graph::Graph g;
    
    g.create_node("input");
    g.create_node("conv1");
    g.create_node("conv2");
    g.create_node("output");
    
    g.connect("input", "conv1");
    g.connect("conv1", "conv2");
    g.connect("conv2", "output");
    
    std::vector<std::shared_ptr<graph::Node>> sorted_nodes;
    auto status = g.topological_sort(sorted_nodes);
    
    assert(status == core::Status::SUCCESS);
    assert(sorted_nodes.size() == 4);
    assert(sorted_nodes[0]->name() == "input");
    assert(sorted_nodes[3]->name() == "output");
    
    std::cout << "✓ Topological sort test passed" << std::endl;
}

int main() {
    std::cout << "\n=== Mini-Infer Graph Tests ===" << std::endl;
    
    try {
        test_graph_creation();
        test_graph_connection();
        test_topological_sort();
        
        std::cout << "\n✓ All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed: " << e.what() << std::endl;
        return 1;
    }
}

