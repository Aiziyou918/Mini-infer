#include <gtest/gtest.h>
#include <memory>

#include "mini_infer/graph/graph.h"
#include "mini_infer/graph/constant_folding_pass.h"
#include "mini_infer/core/tensor.h"
#include "mini_infer/operators/generic_operator.h"
#include "mini_infer/operators/plugin_base.h"

using namespace mini_infer;

class ConstantFoldingTest : public ::testing::Test {
protected:
    void SetUp() override {
        graph_ = std::make_shared<graph::Graph>();
    }

    std::shared_ptr<graph::Graph> graph_;
};

TEST_F(ConstantFoldingTest, BasicConstantFolding) {
    // Create a simple graph with constant inputs
    // Shape -> Gather -> output

    // Create input node with constant tensor
    auto input_node = graph_->create_node("input");
    auto input_tensor = std::make_shared<core::Tensor>(
        core::Shape({2, 3, 4}), core::DataType::FLOAT32);
    input_node->set_input_tensors({input_tensor});

    // Create Shape node
    auto shape_node = graph_->create_node("shape");
    auto shape_op = std::make_shared<operators::GenericOperator>("shape", core::OpType::kSHAPE);
    shape_node->set_operator(shape_op);

    // Connect input -> shape
    auto status = graph_->connect("input", "shape");
    EXPECT_EQ(status, core::Status::SUCCESS);

    // Set graph inputs/outputs
    graph_->set_inputs({"input"});
    graph_->set_outputs({"shape"});

    // Apply constant folding pass
    graph::ConstantFoldingPass pass;
    int num_modifications = 0;
    status = pass.apply(graph_.get(), num_modifications);

    EXPECT_EQ(status, core::Status::SUCCESS);

    // Check if constants were registered (using node ID)
    auto shape_node_ptr = graph_->get_node("shape");
    ASSERT_NE(shape_node_ptr, nullptr);
    EXPECT_TRUE(graph_->is_constant(shape_node_ptr->id()));

    std::cout << "Constant folding test completed. Modifications: "
              << num_modifications << std::endl;
}

TEST_F(ConstantFoldingTest, CheckConstantStorage) {
    // Test constant storage API
    auto tensor = std::make_shared<core::Tensor>(
        core::Shape({1, 2, 3}), core::DataType::FLOAT32);

    // Create a test node
    auto test_node = graph_->create_node("test_constant");
    size_t node_id = test_node->id();

    // Set constant using node ID
    graph_->set_constant(node_id, tensor);

    // Check if constant exists
    EXPECT_TRUE(graph_->is_constant(node_id));

    // Get constant
    auto retrieved = graph_->get_constant(node_id);
    EXPECT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->shape(), tensor->shape());

    // Check non-existent constant
    EXPECT_FALSE(graph_->is_constant(9999));
    EXPECT_EQ(graph_->get_constant(9999), nullptr);
}

TEST_F(ConstantFoldingTest, MultipleConstants) {
    // Test multiple constants
    for (int i = 0; i < 5; ++i) {
        auto tensor = std::make_shared<core::Tensor>(
            core::Shape({i + 1}), core::DataType::FLOAT32);
        auto node = graph_->create_node("const_" + std::to_string(i));
        graph_->set_constant(node->id(), tensor);
    }

    // Verify all constants
    for (int i = 0; i < 5; ++i) {
        auto node = graph_->get_node("const_" + std::to_string(i));
        ASSERT_NE(node, nullptr);
        EXPECT_TRUE(graph_->is_constant(node->id()));
    }

    // Check constants map size
    EXPECT_EQ(graph_->constants().size(), 5);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
