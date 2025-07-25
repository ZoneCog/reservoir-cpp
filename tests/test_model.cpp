/**
 * @file test_model.cpp
 * @brief Test cases for Model class
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "reservoircpp/model.hpp"
#include "reservoircpp/reservoir.hpp"
#include "reservoircpp/readout.hpp"
#include "reservoircpp/matrix_generators.hpp"

using namespace reservoircpp;

TEST_CASE("Model construction", "[model]") {
    SECTION("Empty model") {
        Model model;
        REQUIRE(model.is_empty());
        REQUIRE(model.get_nodes().empty());
        REQUIRE(model.get_edges().empty());
        REQUIRE(model.get_input_nodes().empty());
        REQUIRE(model.get_output_nodes().empty());
    }
    
    SECTION("Model with nodes but no edges") {
        auto node1 = std::make_shared<Node>("node1");
        auto node2 = std::make_shared<Node>("node2");
        
        Model model({node1, node2});
        
        REQUIRE_FALSE(model.is_empty());
        REQUIRE(model.get_nodes().size() == 2);
        REQUIRE(model.get_edges().empty());
        REQUIRE(model.has_node("node1"));
        REQUIRE(model.has_node("node2"));
        REQUIRE_FALSE(model.has_node("node3"));
        
        // With no edges, all nodes are both input and output
        REQUIRE(model.get_input_nodes().size() == 2);
        REQUIRE(model.get_output_nodes().size() == 2);
    }
    
    SECTION("Model with nodes and edges") {
        auto node1 = std::make_shared<Node>("node1");
        auto node2 = std::make_shared<Node>("node2");
        auto node3 = std::make_shared<Node>("node3");
        
        std::vector<NodePtr> nodes = {node1, node2, node3};
        std::vector<Edge> edges = {{node1, node2}, {node2, node3}};
        
        Model model(nodes, edges);
        
        REQUIRE_FALSE(model.is_empty());
        REQUIRE(model.get_nodes().size() == 3);
        REQUIRE(model.get_edges().size() == 2);
        
        // node1 is input, node3 is output, node2 is intermediate
        REQUIRE(model.get_input_nodes().size() == 1);
        REQUIRE(model.get_output_nodes().size() == 1);
        REQUIRE(model.get_input_nodes()[0]->name() == "node1");
        REQUIRE(model.get_output_nodes()[0]->name() == "node3");
    }
}

TEST_CASE("Model node management", "[model]") {
    Model model;
    
    SECTION("Add nodes") {
        auto node1 = std::make_shared<Node>("node1");
        auto node2 = std::make_shared<Node>("node2");
        
        model.add_node(node1);
        REQUIRE(model.has_node("node1"));
        REQUIRE(model.get_nodes().size() == 1);
        
        model.add_node(node2);
        REQUIRE(model.has_node("node2"));
        REQUIRE(model.get_nodes().size() == 2);
        
        // Try to add duplicate
        REQUIRE_THROWS_AS(model.add_node(node1), std::invalid_argument);
    }
    
    SECTION("Add edges") {
        auto node1 = std::make_shared<Node>("node1");
        auto node2 = std::make_shared<Node>("node2");
        
        model.add_node(node1);
        model.add_node(node2);
        
        model.add_edge(node1, node2);
        REQUIRE(model.get_edges().size() == 1);
        
        // Try to add edge with node not in model
        auto node3 = std::make_shared<Node>("node3");
        REQUIRE_THROWS_AS(model.add_edge(node1, node3), std::invalid_argument);
    }
    
    SECTION("Get node by name") {
        auto node1 = std::make_shared<Node>("test_node");
        model.add_node(node1);
        
        auto retrieved = model.get_node("test_node");
        REQUIRE(retrieved->name() == "test_node");
        
        REQUIRE_THROWS_AS(model.get_node("nonexistent"), std::invalid_argument);
    }
}

TEST_CASE("Model connection operators", "[model]") {
    SECTION("Node to node connection") {
        auto node1 = std::make_shared<Node>("node1");
        auto node2 = std::make_shared<Node>("node2");
        
        auto model = node1 >> node2;
        
        REQUIRE(model->get_nodes().size() == 2);
        REQUIRE(model->get_edges().size() == 1);
        REQUIRE(model->has_node("node1"));
        REQUIRE(model->has_node("node2"));
        
        auto edges = model->get_edges();
        REQUIRE(edges[0].first->name() == "node1");
        REQUIRE(edges[0].second->name() == "node2");
    }
    
    SECTION("Chain of connections") {
        auto node1 = std::make_shared<Node>("node1");
        auto node2 = std::make_shared<Node>("node2");
        auto node3 = std::make_shared<Node>("node3");
        
        auto model1 = node1 >> node2;
        auto model2 = model1 >> node3;
        
        REQUIRE(model2->get_nodes().size() == 3);
        REQUIRE(model2->has_node("node1"));
        REQUIRE(model2->has_node("node2"));
        REQUIRE(model2->has_node("node3"));
    }
}

TEST_CASE("Model topological sort", "[model]") {
    auto node1 = std::make_shared<Node>("A");
    auto node2 = std::make_shared<Node>("B");
    auto node3 = std::make_shared<Node>("C");
    auto node4 = std::make_shared<Node>("D");
    
    SECTION("Linear chain") {
        // A -> B -> C -> D
        std::vector<NodePtr> nodes = {node1, node2, node3, node4};
        std::vector<Edge> edges = {{node1, node2}, {node2, node3}, {node3, node4}};
        
        Model model(nodes, edges);
        auto sorted_nodes = model.get_nodes();
        
        REQUIRE(sorted_nodes.size() == 4);
        // Should be in topological order
        REQUIRE(sorted_nodes[0]->name() == "A");
        REQUIRE(sorted_nodes[1]->name() == "B");
        REQUIRE(sorted_nodes[2]->name() == "C");
        REQUIRE(sorted_nodes[3]->name() == "D");
    }
    
    SECTION("Complex DAG") {
        // A -> B, A -> C, B -> D, C -> D
        std::vector<NodePtr> nodes = {node1, node2, node3, node4};
        std::vector<Edge> edges = {{node1, node2}, {node1, node3}, {node2, node4}, {node3, node4}};
        
        Model model(nodes, edges);
        auto sorted_nodes = model.get_nodes();
        
        REQUIRE(sorted_nodes.size() == 4);
        // A should be first, D should be last
        REQUIRE(sorted_nodes[0]->name() == "A");
        REQUIRE(sorted_nodes[3]->name() == "D");
    }
}

TEST_CASE("Model forward pass", "[model]") {
    SECTION("Simple chain") {
        auto node1 = std::make_shared<Node>("input");
        auto node2 = std::make_shared<Node>("output");
        
        auto model = node1 >> node2;
        
        Matrix input = Matrix::Random(5, 3);
        
        // Initialize and run forward pass
        model->initialize(&input);
        Matrix output = model->forward(input);
        
        // Output should have valid dimensions
        REQUIRE(output.rows() > 0);
        REQUIRE(output.cols() > 0);
    }
}

TEST_CASE("Model reset and state management", "[model]") {
    auto node1 = std::make_shared<Node>("node1");
    auto node2 = std::make_shared<Node>("node2");
    
    auto model = node1 >> node2;
    
    Matrix input = Matrix::Random(5, 3);
    model->initialize(&input);
    
    // Run forward pass to set states
    model->forward(input);
    
    // Reset should clear all node states
    model->reset();
    
    // After reset, all nodes should have zero states
    for (const auto& node : model->get_nodes()) {
        Vector state = node->get_state();
        if (state.size() > 0) {
            REQUIRE(state.norm() == Catch::Approx(0.0).margin(1e-10));
        }
    }
}

TEST_CASE("Model copy", "[model]") {
    auto node1 = std::make_shared<Node>("node1");
    auto node2 = std::make_shared<Node>("node2");
    
    auto model = node1 >> node2;
    auto model_copy = std::static_pointer_cast<Model>(model->copy("model_copy"));
    
    REQUIRE(model_copy->name() == "model_copy");
    REQUIRE(model_copy->get_nodes().size() == model->get_nodes().size());
    REQUIRE(model_copy->get_edges().size() == model->get_edges().size());
    
    // Nodes should be different instances but have similar names (with _copy suffix)
    REQUIRE(model_copy->has_node("node1_copy"));
    REQUIRE(model_copy->has_node("node2_copy"));
    
    // But they should be different objects
    REQUIRE(model_copy->get_node("node1_copy") != model->get_node("node1"));
}

TEST_CASE("Model error handling", "[model]") {
    SECTION("Null nodes") {
        REQUIRE_THROWS_AS(Model({nullptr}), std::invalid_argument);
    }
    
    SECTION("Null edges") {
        auto node1 = std::make_shared<Node>("node1");
        REQUIRE_THROWS_AS(Model({node1}, {{nullptr, node1}}), std::invalid_argument);
    }
    
    SECTION("Edge with unknown node") {
        auto node1 = std::make_shared<Node>("node1");
        auto node2 = std::make_shared<Node>("node2");
        auto node3 = std::make_shared<Node>("node3");
        
        // Try to create edge with node not in nodes list
        REQUIRE_THROWS_AS(Model({node1, node2}, {{node1, node3}}), std::invalid_argument);
    }
    
    SECTION("Cycle detection") {
        auto node1 = std::make_shared<Node>("node1");
        auto node2 = std::make_shared<Node>("node2");
        
        // Create a cycle: A -> B -> A
        std::vector<NodePtr> nodes = {node1, node2};
        std::vector<Edge> edges = {{node1, node2}, {node2, node1}};
        
        REQUIRE_THROWS_AS(Model(nodes, edges), std::runtime_error);
    }
}