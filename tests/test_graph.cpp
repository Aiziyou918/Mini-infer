#include "mini_infer/graph/graph.h"
#include "mini_infer/graph/node.h"
#include <gtest/gtest.h>

using namespace mini_infer;

class GraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        graph_ = std::make_unique<graph::Graph>();
    }

    void TearDown() override {
        graph_.reset();
    }

    std::unique_ptr<graph::Graph> graph_;
};

// ============================================================================
// Node Creation Tests
// ============================================================================

TEST_F(GraphTest, NodeCreation) {
    auto node1 = graph_->create_node("input");
    auto node2 = graph_->create_node("conv");
    auto node3 = graph_->create_node("output");
    
    ASSERT_NE(node1, nullptr);
    ASSERT_NE(node2, nullptr);
    ASSERT_NE(node3, nullptr);
    
    EXPECT_EQ(node1->name(), "input");
    EXPECT_EQ(node2->name(), "conv");
    EXPECT_EQ(node3->name(), "output");
}

TEST_F(GraphTest, MultipleNodeCreation) {
    for (int i = 0; i < 10; ++i) {
        std::string name = "node_" + std::to_string(i);
        auto node = graph_->create_node(name);
        ASSERT_NE(node, nullptr);
        EXPECT_EQ(node->name(), name);
    }
}

// ============================================================================
// Graph Connection Tests
// ============================================================================

TEST_F(GraphTest, BasicConnection) {
    auto node1 = graph_->create_node("input");
    auto node2 = graph_->create_node("conv");
    auto node3 = graph_->create_node("output");
    
    auto status = graph_->connect("input", "conv");
    EXPECT_EQ(status, core::Status::SUCCESS);
    
    status = graph_->connect("conv", "output");
    EXPECT_EQ(status, core::Status::SUCCESS);
}

TEST_F(GraphTest, ConnectionVerification) {
    auto node1 = graph_->create_node("input");
    auto node2 = graph_->create_node("conv");
    auto node3 = graph_->create_node("output");
    
    graph_->connect("input", "conv");
    graph_->connect("conv", "output");
    
    EXPECT_EQ(node1->outputs().size(), 1);
    EXPECT_EQ(node2->inputs().size(), 1);
    EXPECT_EQ(node2->outputs().size(), 1);
    EXPECT_EQ(node3->inputs().size(), 1);
}

TEST_F(GraphTest, MultipleInputsConnection) {
    graph_->create_node("input1");
    graph_->create_node("input2");
    auto node3 = graph_->create_node("concat");
    
    graph_->connect("input1", "concat");
    graph_->connect("input2", "concat");
    
    EXPECT_EQ(node3->inputs().size(), 2);
}

TEST_F(GraphTest, MultipleOutputsConnection) {
    auto node1 = graph_->create_node("input");
    graph_->create_node("branch1");
    graph_->create_node("branch2");
    
    graph_->connect("input", "branch1");
    graph_->connect("input", "branch2");
    
    EXPECT_EQ(node1->outputs().size(), 2);
}

// ============================================================================
// Topological Sort Tests
// ============================================================================

TEST_F(GraphTest, LinearTopologicalSort) {
    graph_->create_node("input");
    graph_->create_node("conv1");
    graph_->create_node("conv2");
    graph_->create_node("output");
    
    graph_->connect("input", "conv1");
    graph_->connect("conv1", "conv2");
    graph_->connect("conv2", "output");
    
    std::vector<std::shared_ptr<graph::Node>> sorted_nodes;
    auto status = graph_->topological_sort(sorted_nodes);
    
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(sorted_nodes.size(), 4);
    EXPECT_EQ(sorted_nodes[0]->name(), "input");
    EXPECT_EQ(sorted_nodes[1]->name(), "conv1");
    EXPECT_EQ(sorted_nodes[2]->name(), "conv2");
    EXPECT_EQ(sorted_nodes[3]->name(), "output");
}

TEST_F(GraphTest, BranchTopologicalSort) {
    graph_->create_node("input");
    graph_->create_node("branch1");
    graph_->create_node("branch2");
    graph_->create_node("merge");
    
    graph_->connect("input", "branch1");
    graph_->connect("input", "branch2");
    graph_->connect("branch1", "merge");
    graph_->connect("branch2", "merge");
    
    std::vector<std::shared_ptr<graph::Node>> sorted_nodes;
    auto status = graph_->topological_sort(sorted_nodes);
    
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(sorted_nodes.size(), 4);
    EXPECT_EQ(sorted_nodes[0]->name(), "input");
    EXPECT_EQ(sorted_nodes[3]->name(), "merge");
}

TEST_F(GraphTest, SingleNodeTopologicalSort) {
    graph_->create_node("single");
    
    std::vector<std::shared_ptr<graph::Node>> sorted_nodes;
    auto status = graph_->topological_sort(sorted_nodes);
    
    EXPECT_EQ(status, core::Status::SUCCESS);
    ASSERT_EQ(sorted_nodes.size(), 1);
    EXPECT_EQ(sorted_nodes[0]->name(), "single");
}

// ============================================================================
// Graph Validation Tests
// ============================================================================

TEST_F(GraphTest, ValidateValidGraph) {
    graph_->create_node("input");
    graph_->create_node("conv");
    graph_->create_node("output");
    
    graph_->connect("input", "conv");
    graph_->connect("conv", "output");
    
    graph_->set_inputs({"input"});
    graph_->set_outputs({"output"});
    
    auto status = graph_->validate();
    EXPECT_EQ(status, core::Status::SUCCESS);
}

// ============================================================================
// Input/Output Tests
// ============================================================================

TEST_F(GraphTest, SetInputsAndOutputs) {
    graph_->create_node("input1");
    graph_->create_node("input2");
    graph_->create_node("output");
    
    graph_->set_inputs({"input1", "input2"});
    graph_->set_outputs({"output"});
    
    auto inputs = graph_->inputs();
    auto outputs = graph_->outputs();
    
    EXPECT_EQ(inputs.size(), 2);
    EXPECT_EQ(outputs.size(), 1);
}

