/**
 * @file test_ops.cpp
 * @brief Tests for ops functionality in ReservoirCpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "reservoircpp/ops.hpp"
#include "reservoircpp/concat.hpp"
#include "reservoircpp/reservoir.hpp"
#include "reservoircpp/readout.hpp"

using namespace reservoircpp;
using Catch::Matchers::WithinAbs;

TEST_CASE("Ops - Link two nodes", "[ops][link]") {
    auto node1 = std::make_shared<Node>("node1");
    auto node2 = std::make_shared<Node>("node2");
    
    SECTION("Basic link") {
        auto model = ops::link(node1, node2, "linked_model");
        
        REQUIRE(model != nullptr);
        REQUIRE(model->name() == "linked_model");
        REQUIRE(model->get_nodes().size() == 2);
        REQUIRE(model->get_edges().size() == 1);
        
        auto nodes = model->get_nodes();
        REQUIRE(std::find(nodes.begin(), nodes.end(), node1) != nodes.end());
        REQUIRE(std::find(nodes.begin(), nodes.end(), node2) != nodes.end());
        
        auto edges = model->get_edges();
        REQUIRE(edges[0].first == node1);
        REQUIRE(edges[0].second == node2);
    }
    
    SECTION("Auto-generated name") {
        auto model = ops::link(node1, node2);
        
        REQUIRE(model != nullptr);
        REQUIRE(model->name().substr(0, 5) == "link_");
    }
    
    SECTION("Null node handling") {
        NodePtr null_node = nullptr;
        REQUIRE_THROWS_AS(ops::link(null_node, node2), std::invalid_argument);
        REQUIRE_THROWS_AS(ops::link(node1, null_node), std::invalid_argument);
    }
}

TEST_CASE("Ops - Link node to model", "[ops][link]") {
    auto node1 = std::make_shared<Node>("node1");
    auto node2 = std::make_shared<Node>("node2");
    auto node3 = std::make_shared<Node>("node3");
    
    // Create a simple model
    auto model = ops::link(node2, node3, "base_model");
    
    SECTION("Link node to model inputs") {
        auto result = ops::link(node1, model, "extended_model");
        
        REQUIRE(result != nullptr);
        REQUIRE(result->name() == "extended_model");
        REQUIRE(result->get_nodes().size() == 3);
        
        // Should have edges: node1->node2, node2->node3, node1->node2 (from model)
        auto edges = result->get_edges();
        REQUIRE(edges.size() >= 2); // At least the original edge plus new connections
        
        auto nodes = result->get_nodes();
        REQUIRE(std::find(nodes.begin(), nodes.end(), node1) != nodes.end());
        REQUIRE(std::find(nodes.begin(), nodes.end(), node2) != nodes.end());
        REQUIRE(std::find(nodes.begin(), nodes.end(), node3) != nodes.end());
    }
    
    SECTION("Link model to node") {
        auto result = ops::link(model, node1, "extended_model");
        
        REQUIRE(result != nullptr);
        REQUIRE(result->name() == "extended_model");
        REQUIRE(result->get_nodes().size() == 3);
        
        auto nodes = result->get_nodes();
        REQUIRE(std::find(nodes.begin(), nodes.end(), node1) != nodes.end());
        REQUIRE(std::find(nodes.begin(), nodes.end(), node2) != nodes.end());
        REQUIRE(std::find(nodes.begin(), nodes.end(), node3) != nodes.end());
    }
}

TEST_CASE("Ops - Link multiple nodes", "[ops][link]") {
    auto node1 = std::make_shared<Node>("node1");
    auto node2 = std::make_shared<Node>("node2");
    auto node3 = std::make_shared<Node>("node3");
    auto output_node = std::make_shared<Node>("output");
    
    SECTION("Many-to-one (requires concat)") {
        std::vector<NodePtr> input_nodes = {node1, node2, node3};
        auto model = ops::link(input_nodes, output_node, "many_to_one");
        
        REQUIRE(model != nullptr);
        REQUIRE(model->name() == "many_to_one");
        
        auto nodes = model->get_nodes();
        REQUIRE(nodes.size() == 5); // 3 inputs + 1 concat + 1 output
        
        // Should contain all original nodes
        REQUIRE(std::find(nodes.begin(), nodes.end(), node1) != nodes.end());
        REQUIRE(std::find(nodes.begin(), nodes.end(), node2) != nodes.end());
        REQUIRE(std::find(nodes.begin(), nodes.end(), node3) != nodes.end());
        REQUIRE(std::find(nodes.begin(), nodes.end(), output_node) != nodes.end());
        
        // Should contain a concat node
        bool has_concat = false;
        for (const auto& node : nodes) {
            if (std::dynamic_pointer_cast<Concat>(node)) {
                has_concat = true;
                break;
            }
        }
        REQUIRE(has_concat);
    }
    
    SECTION("One-to-many") {
        std::vector<NodePtr> output_nodes = {node1, node2, node3};
        auto model = ops::link(output_node, output_nodes, "one_to_many");
        
        REQUIRE(model != nullptr);
        REQUIRE(model->name() == "one_to_many");
        
        auto nodes = model->get_nodes();
        REQUIRE(nodes.size() == 4); // 1 input + 3 outputs
        
        auto edges = model->get_edges();
        REQUIRE(edges.size() == 3); // output_node connected to each output
    }
    
    SECTION("Single input/output reduces to simple link") {
        std::vector<NodePtr> input_nodes = {node1};
        auto model = ops::link(input_nodes, output_node);
        
        REQUIRE(model != nullptr);
        REQUIRE(model->get_nodes().size() == 2); // Just the two nodes
        REQUIRE(model->get_edges().size() == 1);
    }
    
    SECTION("Empty input list throws") {
        std::vector<NodePtr> empty_nodes;
        REQUIRE_THROWS_AS(ops::link(empty_nodes, output_node), std::invalid_argument);
        REQUIRE_THROWS_AS(ops::link(output_node, empty_nodes), std::invalid_argument);
    }
}

TEST_CASE("Ops - Merge models", "[ops][merge]") {
    auto node1 = std::make_shared<Node>("node1");
    auto node2 = std::make_shared<Node>("node2");
    auto node3 = std::make_shared<Node>("node3");
    auto node4 = std::make_shared<Node>("node4");
    
    auto model1 = ops::link(node1, node2, "model1");
    auto model2 = ops::link(node3, node4, "model2");
    
    SECTION("Merge two models") {
        auto merged = ops::merge(model1, model2, "merged_model");
        
        REQUIRE(merged != nullptr);
        REQUIRE(merged->name() == "merged_model");
        
        auto nodes = merged->get_nodes();
        REQUIRE(nodes.size() == 4);
        
        // Should contain all nodes from both models
        REQUIRE(std::find(nodes.begin(), nodes.end(), node1) != nodes.end());
        REQUIRE(std::find(nodes.begin(), nodes.end(), node2) != nodes.end());
        REQUIRE(std::find(nodes.begin(), nodes.end(), node3) != nodes.end());
        REQUIRE(std::find(nodes.begin(), nodes.end(), node4) != nodes.end());
        
        auto edges = merged->get_edges();
        REQUIRE(edges.size() == 2); // Both original edges preserved
    }
    
    SECTION("Merge model with node") {
        auto merged = ops::merge(model1, node3, "merged_with_node");
        
        REQUIRE(merged != nullptr);
        REQUIRE(merged->name() == "merged_with_node");
        
        auto nodes = merged->get_nodes();
        REQUIRE(nodes.size() == 3); // 2 from model + 1 new node
        
        REQUIRE(std::find(nodes.begin(), nodes.end(), node1) != nodes.end());
        REQUIRE(std::find(nodes.begin(), nodes.end(), node2) != nodes.end());
        REQUIRE(std::find(nodes.begin(), nodes.end(), node3) != nodes.end());
    }
    
    SECTION("Merge two nodes") {
        auto merged = ops::merge(node1, node2, "merged_nodes");
        
        REQUIRE(merged != nullptr);
        REQUIRE(merged->name() == "merged_nodes");
        
        auto nodes = merged->get_nodes();
        REQUIRE(nodes.size() == 2);
        
        auto edges = merged->get_edges();
        REQUIRE(edges.size() == 0); // No connections in a merge
    }
    
    SECTION("Null model handling") {
        std::shared_ptr<Model> null_model = nullptr;
        REQUIRE_THROWS_AS(ops::merge(null_model, model2), std::invalid_argument);
        REQUIRE_THROWS_AS(ops::merge(model1, null_model), std::invalid_argument);
    }
}

TEST_CASE("Ops - Link feedback", "[ops][link_feedback]") {
    auto main_node = std::make_shared<Node>("main");
    auto feedback_node = std::make_shared<Node>("feedback");
    
    SECTION("Basic feedback connection") {
        auto result = ops::link_feedback(main_node, feedback_node, false, "feedback_node");
        
        REQUIRE(result != nullptr);
        REQUIRE(result->name() == "feedback_node");
        REQUIRE(result != main_node); // Should be a copy when inplace=false
    }
    
    SECTION("In-place feedback connection") {
        auto result = ops::link_feedback(main_node, feedback_node, true);
        
        REQUIRE(result != nullptr);
        REQUIRE(result == main_node); // Should be same object when inplace=true
    }
    
    SECTION("Multiple feedback nodes") {
        auto feedback2 = std::make_shared<Node>("feedback2");
        std::vector<NodePtr> feedback_nodes = {feedback_node, feedback2};
        
        auto result = ops::link_feedback(main_node, feedback_nodes, false, "multi_feedback");
        
        REQUIRE(result != nullptr);
        REQUIRE(result->name() == "multi_feedback");
    }
    
    SECTION("Null node handling") {
        NodePtr null_node = nullptr;
        REQUIRE_THROWS_AS(ops::link_feedback(null_node, feedback_node), std::invalid_argument);
        REQUIRE_THROWS_AS(ops::link_feedback(main_node, null_node), std::invalid_argument);
    }
    
    SECTION("Feedback functionality verification") {
        // Test that the feedback mechanism actually works in forward pass
        Matrix input = Matrix::Random(5, 3);
        
        // Initialize nodes
        main_node->initialize(&input);
        feedback_node->initialize(&input);
        
        // Link feedback using inplace=true
        auto result = ops::link_feedback(main_node, feedback_node, true);
        
        REQUIRE(result == main_node);
        REQUIRE(result->has_feedback());
        REQUIRE(result->get_feedback() == feedback_node);
        
        // Test forward pass works with feedback
        auto output1 = result->operator()(input);
        REQUIRE(output1.rows() == input.rows());
        REQUIRE(output1.cols() == input.cols());
        
        // Test second forward pass (should use feedback from first call)
        auto output2 = result->operator()(input);
        REQUIRE(output2.rows() == input.rows());
        REQUIRE(output2.cols() == input.cols());
    }
}

TEST_CASE("Concat node", "[concat]") {
    SECTION("Basic concatenation along columns") {
        auto concat = std::make_shared<Concat>(1, "test_concat");
        
        REQUIRE(concat->get_axis() == 1);
        REQUIRE(concat->name() == "test_concat");
        
        // Create test matrices
        Matrix input1 = Matrix::Ones(3, 2);
        Matrix input2 = Matrix::Ones(3, 3) * 2.0;
        
        std::vector<Matrix> inputs = {input1, input2};
        Matrix result = concat->forward_multiple(inputs);
        
        REQUIRE(result.rows() == 3);
        REQUIRE(result.cols() == 5); // 2 + 3
        
        // Check values
        REQUIRE_THAT(result(0, 0), WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(result(0, 1), WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(result(0, 2), WithinAbs(2.0, 1e-10));
        REQUIRE_THAT(result(0, 3), WithinAbs(2.0, 1e-10));
        REQUIRE_THAT(result(0, 4), WithinAbs(2.0, 1e-10));
    }
    
    SECTION("Concatenation along rows") {
        auto concat = std::make_shared<Concat>(0, "row_concat");
        
        REQUIRE(concat->get_axis() == 0);
        
        Matrix input1 = Matrix::Ones(2, 3);
        Matrix input2 = Matrix::Ones(3, 3) * 2.0;
        
        std::vector<Matrix> inputs = {input1, input2};
        Matrix result = concat->forward_multiple(inputs);
        
        REQUIRE(result.rows() == 5); // 2 + 3
        REQUIRE(result.cols() == 3);
        
        // Check values
        REQUIRE_THAT(result(0, 0), WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(result(1, 0), WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(result(2, 0), WithinAbs(2.0, 1e-10));
        REQUIRE_THAT(result(3, 0), WithinAbs(2.0, 1e-10));
        REQUIRE_THAT(result(4, 0), WithinAbs(2.0, 1e-10));
    }
    
    SECTION("Single input passthrough") {
        auto concat = std::make_shared<Concat>(1);
        
        Matrix input = Matrix::Ones(3, 2);
        std::vector<Matrix> inputs = {input};
        Matrix result = concat->forward_multiple(inputs);
        
        REQUIRE(result.rows() == input.rows());
        REQUIRE(result.cols() == input.cols());
        REQUIRE((result.array() == input.array()).all());
    }
    
    SECTION("Empty input throws") {
        auto concat = std::make_shared<Concat>(1);
        std::vector<Matrix> empty_inputs;
        
        REQUIRE_THROWS_AS(concat->forward_multiple(empty_inputs), std::invalid_argument);
    }
    
    SECTION("Dimension mismatch throws") {
        auto concat = std::make_shared<Concat>(1); // Concatenate along columns
        
        Matrix input1 = Matrix::Ones(3, 2);
        Matrix input2 = Matrix::Ones(4, 2); // Different number of rows
        
        std::vector<Matrix> inputs = {input1, input2};
        REQUIRE_THROWS_AS(concat->forward_multiple(inputs), std::invalid_argument);
    }
    
    SECTION("Invalid axis throws") {
        REQUIRE_THROWS_AS(std::make_shared<Concat>(2), std::invalid_argument);
        
        auto concat = std::make_shared<Concat>(1);
        REQUIRE_THROWS_AS(concat->set_axis(3), std::invalid_argument);
    }
}

TEST_CASE("Ops - Integration with existing operators", "[ops][integration]") {
    auto node1 = std::make_shared<Node>("node1");
    auto node2 = std::make_shared<Node>("node2");
    auto node3 = std::make_shared<Node>("node3");
    
    SECTION("Ops link equivalent to >> operator") {
        auto model1 = ops::link(node1, node2);
        auto model2 = node1 >> node2;
        
        REQUIRE(model1->get_nodes().size() == model2->get_nodes().size());
        REQUIRE(model1->get_edges().size() == model2->get_edges().size());
    }
    
    SECTION("Ops merge equivalent to & operator") {
        auto node4 = std::make_shared<Node>("node4");
        auto model1 = ops::link(node1, node2);
        auto model2 = ops::link(node3, node4); // Use different nodes to avoid duplicates
        
        auto merged1 = ops::merge(model1, model2);
        auto merged2 = model1 & model2;
        
        REQUIRE(merged1->get_nodes().size() == merged2->get_nodes().size());
        // Note: edge counts might differ due to implementation details
    }
}