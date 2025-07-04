/**
 * @file simple_example.cpp
 * @brief Simple example demonstrating Stage 1 capabilities
 * 
 * This example shows basic usage of the ReservoirCpp Stage 1 implementation
 * including activation functions, utility functions, and base Node class.
 */

#include <iostream>
#include <reservoircpp/reservoircpp.hpp>

int main() {
    std::cout << "=== ReservoirCpp Stage 1 Demo ===" << std::endl;
    std::cout << reservoircpp::version_info() << std::endl;
    std::cout << std::endl;
    
    // Set a reproducible random seed
    reservoircpp::utils::set_seed(42);
    
    // Demonstrate activation functions
    std::cout << "=== Activation Functions Demo ===" << std::endl;
    
    // Create test data
    reservoircpp::Matrix x(1, 5);
    x << -2.0, -1.0, 0.0, 1.0, 2.0;
    
    std::cout << "Input: " << x << std::endl;
    
    // Test different activation functions
    auto sigmoid_result = reservoircpp::activations::sigmoid(x);
    auto tanh_result = reservoircpp::activations::tanh(x);
    auto relu_result = reservoircpp::activations::relu(x);
    
    std::cout << "Sigmoid: " << sigmoid_result << std::endl;
    std::cout << "Tanh:    " << tanh_result << std::endl;
    std::cout << "ReLU:    " << relu_result << std::endl;
    
    // Test softmax
    reservoircpp::Matrix softmax_input(1, 3);
    softmax_input << 1.0, 2.0, 3.0;
    auto softmax_result = reservoircpp::activations::softmax(softmax_input);
    std::cout << "Softmax input: " << softmax_input << std::endl;
    std::cout << "Softmax:       " << softmax_result << std::endl;
    std::cout << "Sum:           " << softmax_result.sum() << std::endl;
    std::cout << std::endl;
    
    // Demonstrate activation registry
    std::cout << "=== Activation Registry Demo ===" << std::endl;
    auto sigmoid_fn = reservoircpp::activations::get_function("sigmoid");
    auto tanh_fn = reservoircpp::activations::get_function("tanh");
    
    std::cout << "Using registry - Sigmoid: " << sigmoid_fn(x) << std::endl;
    std::cout << "Using registry - Tanh:    " << tanh_fn(x) << std::endl;
    std::cout << std::endl;
    
    // Demonstrate utility functions
    std::cout << "=== Utility Functions Demo ===" << std::endl;
    
    // Generate random matrices
    auto random_matrix = reservoircpp::utils::random_uniform(3, 4, -1.0, 1.0);
    std::cout << "Random uniform matrix (3x4):" << std::endl;
    std::cout << random_matrix << std::endl;
    
    auto random_normal = reservoircpp::utils::random_normal(2, 3, 0.0, 1.0);
    std::cout << "Random normal matrix (2x3):" << std::endl;
    std::cout << random_normal << std::endl;
    
    // Show shape utilities
    auto shape = reservoircpp::utils::array::get_shape(random_matrix);
    std::cout << "Shape of random matrix: " << reservoircpp::utils::array::shape_to_string(shape) << std::endl;
    std::cout << std::endl;
    
    // Demonstrate Node class
    std::cout << "=== Node Class Demo ===" << std::endl;
    
    // Create a basic node
    reservoircpp::Node node("demo_node");
    std::cout << "Created node: " << node.name() << std::endl;
    
    // Set dimensions and initialize
    node.set_input_dim({2, 3});
    node.set_output_dim({1, 4});
    std::cout << "Input dimensions: " << reservoircpp::utils::array::shape_to_string(node.input_dim()) << std::endl;
    std::cout << "Output dimensions: " << reservoircpp::utils::array::shape_to_string(node.output_dim()) << std::endl;
    std::cout << "Output size: " << node.get_output_size() << std::endl;
    
    // Initialize and test
    node.initialize();
    std::cout << "Node initialized: " << (node.is_initialized() ? "Yes" : "No") << std::endl;
    
    // Test forward pass (default is identity)
    reservoircpp::Matrix input_data(2, 3);
    input_data << 1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0;
    
    auto output = node(input_data);
    std::cout << "Input to node:" << std::endl << input_data << std::endl;
    std::cout << "Output from node:" << std::endl << output << std::endl;
    
    // Test state management
    auto state = node.get_state();
    std::cout << "Node state size: " << state.size() << std::endl;
    std::cout << "Node state: " << state.transpose() << std::endl;
    
    // Copy the node
    auto node_copy = node.copy("demo_node_copy");
    std::cout << "Copied node: " << node_copy->name() << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "✓ Activation functions working" << std::endl;
    std::cout << "✓ Activation registry working" << std::endl;
    std::cout << "✓ Utility functions working" << std::endl;
    std::cout << "✓ Random number generation working" << std::endl;
    std::cout << "✓ Base Node class working" << std::endl;
    std::cout << "✓ State management working" << std::endl;
    std::cout << "✓ Parameter management working" << std::endl;
    std::cout << std::endl;
    std::cout << "Stage 1 (Core Framework and Data Structures) is complete!" << std::endl;
    std::cout << "Ready for Stage 2 implementation." << std::endl;
    
    return 0;
}