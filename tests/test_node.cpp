/**
 * @file test_node.cpp
 * @brief Tests for base Node class
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <reservoircpp/node.hpp>

using namespace reservoircpp;

TEST_CASE("Node construction", "[node]") {
    SECTION("Default constructor") {
        Node node;
        
        REQUIRE(node.name().length() > 0);
        REQUIRE_FALSE(node.is_initialized());
        REQUIRE(node.is_trainable());
        REQUIRE(node.input_dim().empty());
        REQUIRE(node.output_dim().empty());
        REQUIRE(node.get_input_size() == 0);
        REQUIRE(node.get_output_size() == 0);
    }
    
    SECTION("Constructor with name") {
        Node node("test_node");
        
        REQUIRE(node.name() == "test_node");
        REQUIRE_FALSE(node.is_initialized());
        REQUIRE(node.is_trainable());
    }
    
    SECTION("Constructor with parameters") {
        ParameterMap params;
        params["param1"] = 42;
        params["param2"] = 3.14;
        
        ParameterMap hypers;
        hypers["hyper1"] = std::string("test");
        
        Node node("test_node", params, hypers);
        
        REQUIRE(node.name() == "test_node");
        REQUIRE(node.has_param("param1"));
        REQUIRE(node.has_param("param2"));
        REQUIRE(node.has_param("hyper1"));
        
        REQUIRE(std::any_cast<int>(node.get_param("param1")) == 42);
        REQUIRE(std::any_cast<double>(node.get_param("param2")) == Catch::Approx(3.14));
        REQUIRE(std::any_cast<std::string>(node.get_param("hyper1")) == "test");
    }
}

TEST_CASE("Node parameter management", "[node]") {
    Node node("test_node");
    
    SECTION("Set and get parameters") {
        // First add the parameters to the node's parameter map
        ParameterMap params;
        params["param1"] = 100;
        params["param2"] = 2.71;
        
        Node node_with_params("test_node", params);
        
        REQUIRE(node_with_params.has_param("param1"));
        REQUIRE(node_with_params.has_param("param2"));
        REQUIRE_FALSE(node_with_params.has_param("nonexistent"));
        
        REQUIRE(std::any_cast<int>(node_with_params.get_param("param1")) == 100);
        REQUIRE(std::any_cast<double>(node_with_params.get_param("param2")) == Catch::Approx(2.71));
        
        // Test modifying existing parameters
        node_with_params.set_param("param1", 200);
        REQUIRE(std::any_cast<int>(node_with_params.get_param("param1")) == 200);
    }
    
    SECTION("Get nonexistent parameter throws") {
        REQUIRE_THROWS_AS(node.get_param("nonexistent"), std::invalid_argument);
    }
    
    SECTION("Set nonexistent parameter throws") {
        REQUIRE_THROWS_AS(node.set_param("nonexistent", 42), std::invalid_argument);
    }
    
    SECTION("Get parameter names") {
        ParameterMap params;
        params["param1"] = 1;
        params["param2"] = 2;
        
        Node node_with_params("test_node", params);
        
        auto names = node_with_params.get_param_names();
        REQUIRE(names.size() == 2);
        REQUIRE(std::find(names.begin(), names.end(), "param1") != names.end());
        REQUIRE(std::find(names.begin(), names.end(), "param2") != names.end());
    }
}

TEST_CASE("Node dimensions", "[node]") {
    Node node("test_node");
    
    SECTION("Set dimensions before initialization") {
        node.set_input_dim({10, 5});
        node.set_output_dim({3, 2});
        
        REQUIRE(node.input_dim() == Shape({10, 5}));
        REQUIRE(node.output_dim() == Shape({3, 2}));
        REQUIRE(node.get_input_size() == 50);
        REQUIRE(node.get_output_size() == 6);
    }
    
    SECTION("Cannot set dimensions after initialization") {
        Matrix x(2, 3);
        node.initialize(&x);
        
        REQUIRE_THROWS_AS(node.set_input_dim({5, 5}), std::runtime_error);
        REQUIRE_THROWS_AS(node.set_output_dim({4, 4}), std::runtime_error);
    }
    
    SECTION("Dimensions from initialization") {
        Matrix x(3, 4);
        Matrix y(2, 5);
        
        node.initialize(&x, &y);
        
        REQUIRE(node.input_dim() == Shape({3, 4}));
        REQUIRE(node.output_dim() == Shape({2, 5}));
        REQUIRE(node.get_input_size() == 12);
        REQUIRE(node.get_output_size() == 10);
    }
}

TEST_CASE("Node initialization", "[node]") {
    SECTION("Manual initialization") {
        Node node("test_node");
        Matrix x(2, 3);
        
        REQUIRE_FALSE(node.is_initialized());
        
        node.initialize(&x);
        
        REQUIRE(node.is_initialized());
        REQUIRE(node.input_dim() == Shape({2, 3}));
    }
    
    SECTION("Automatic initialization on first call") {
        Node node("test_node");
        Matrix x(2, 3);
        
        REQUIRE_FALSE(node.is_initialized());
        
        Matrix output = node(x);
        
        REQUIRE(node.is_initialized());
        REQUIRE(node.input_dim() == Shape({2, 3}));
        REQUIRE(output.rows() == x.rows());
        REQUIRE(output.cols() == x.cols());
    }
    
    SECTION("Multiple initializations are safe") {
        Node node("test_node");
        Matrix x1(2, 3);
        Matrix x2(4, 5);
        
        node.initialize(&x1);
        REQUIRE(node.input_dim() == Shape({2, 3}));
        
        // Second initialization should be ignored
        node.initialize(&x2);
        REQUIRE(node.input_dim() == Shape({2, 3}));
    }
}

TEST_CASE("Node state management", "[node]") {
    Node node("test_node");
    
    SECTION("Default state") {
        node.set_output_dim({5});
        node.initialize();
        
        Vector state = node.get_state();
        REQUIRE(state.size() == 5);
        
        // Default state should be zero
        for (int i = 0; i < state.size(); ++i) {
            REQUIRE(state[i] == Catch::Approx(0.0));
        }
    }
    
    SECTION("Set and get state") {
        node.set_output_dim({3});
        node.initialize();
        
        Vector new_state(3);
        new_state << 1.0, 2.0, 3.0;
        
        node.set_state(new_state);
        Vector retrieved_state = node.get_state();
        
        REQUIRE(retrieved_state.size() == 3);
        REQUIRE(retrieved_state[0] == Catch::Approx(1.0));
        REQUIRE(retrieved_state[1] == Catch::Approx(2.0));
        REQUIRE(retrieved_state[2] == Catch::Approx(3.0));
    }
    
    SECTION("Reset state") {
        node.set_output_dim({3});
        node.initialize();
        
        Vector new_state(3);
        new_state << 1.0, 2.0, 3.0;
        node.set_state(new_state);
        
        // Reset to zero
        node.reset();
        Vector state = node.get_state();
        
        for (int i = 0; i < state.size(); ++i) {
            REQUIRE(state[i] == Catch::Approx(0.0));
        }
    }
    
    SECTION("Reset to specific state") {
        node.set_output_dim({3});
        node.initialize();
        
        Vector reset_state(3);
        reset_state << 5.0, 6.0, 7.0;
        
        node.reset(&reset_state);
        Vector state = node.get_state();
        
        REQUIRE(state[0] == Catch::Approx(5.0));
        REQUIRE(state[1] == Catch::Approx(6.0));
        REQUIRE(state[2] == Catch::Approx(7.0));
    }
    
    SECTION("State validation") {
        node.set_output_dim({3});
        node.initialize();
        
        Vector wrong_size_state(5);
        wrong_size_state << 1.0, 2.0, 3.0, 4.0, 5.0;
        
        REQUIRE_THROWS_AS(node.set_state(wrong_size_state), std::invalid_argument);
        REQUIRE_THROWS_AS(node.reset(&wrong_size_state), std::invalid_argument);
    }
}

TEST_CASE("Node forward pass", "[node]") {
    Node node("test_node");
    
    SECTION("Default forward pass is identity") {
        Matrix input(2, 3);
        input << 1.0, 2.0, 3.0,
                 4.0, 5.0, 6.0;
        
        Matrix output = node(input);
        
        REQUIRE(output.rows() == input.rows());
        REQUIRE(output.cols() == input.cols());
        
        for (int i = 0; i < input.rows(); ++i) {
            for (int j = 0; j < input.cols(); ++j) {
                REQUIRE(output(i, j) == Catch::Approx(input(i, j)));
            }
        }
    }
    
    SECTION("Forward pass initializes node") {
        Node node("test_node");
        Matrix input(2, 3);
        
        REQUIRE_FALSE(node.is_initialized());
        
        node(input);
        
        REQUIRE(node.is_initialized());
        REQUIRE(node.input_dim() == Shape({2, 3}));
    }
}

TEST_CASE("Node copy", "[node]") {
    SECTION("Copy node") {
        ParameterMap params;
        params["param1"] = 42;
        
        Node original("original", params);
        original.set_input_dim({2, 3});
        original.set_output_dim({1, 4});
        
        Matrix x(2, 3);
        original.initialize(&x);
        
        auto copy = original.copy("copy");
        
        REQUIRE(copy->name() == "copy");
        REQUIRE(copy->has_param("param1"));
        REQUIRE(std::any_cast<int>(copy->get_param("param1")) == 42);
        REQUIRE(copy->input_dim() == Shape({2, 3}));
        REQUIRE(copy->output_dim() == Shape({1, 4}));
        REQUIRE(copy->is_initialized());
    }
    
    SECTION("Copy with auto-generated name") {
        Node original("original");
        auto copy = original.copy();
        
        REQUIRE(copy->name() != "original");
        REQUIRE(copy->name().length() > 0);
    }
}

TEST_CASE("Node zero state", "[node]") {
    Node node("test_node");
    
    SECTION("Zero state for 1D output") {
        node.set_output_dim({5});
        
        Vector zero_state = node.zero_state();
        REQUIRE(zero_state.size() == 5);
        
        for (int i = 0; i < zero_state.size(); ++i) {
            REQUIRE(zero_state[i] == Catch::Approx(0.0));
        }
    }
    
    SECTION("Zero state for 2D output") {
        node.set_output_dim({3, 4});
        
        Vector zero_state = node.zero_state();
        REQUIRE(zero_state.size() == 12);
        
        for (int i = 0; i < zero_state.size(); ++i) {
            REQUIRE(zero_state[i] == Catch::Approx(0.0));
        }
    }
}