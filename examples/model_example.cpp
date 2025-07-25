/**
 * @file model_example.cpp
 * @brief Example demonstrating Model class usage
 */

#include <iostream>
#include <memory>
#include "reservoircpp/reservoircpp.hpp"

using namespace reservoircpp;

int main() {
    std::cout << "ReservoirCpp Model Example\n";
    std::cout << "==========================\n\n";
    
    // Create some nodes
    auto input_node = std::make_shared<Node>("input");
    auto hidden_node = std::make_shared<Node>("hidden");
    auto output_node = std::make_shared<Node>("output");
    
    std::cout << "1. Creating individual nodes:\n";
    std::cout << "   - Input node: " << input_node->name() << "\n";
    std::cout << "   - Hidden node: " << hidden_node->name() << "\n";
    std::cout << "   - Output node: " << output_node->name() << "\n\n";
    
    // Create a simple chain model using the >> operator
    std::cout << "2. Creating a linear chain model using >> operator:\n";
    auto chain_model = input_node >> hidden_node >> output_node;
    std::cout << "   Model created with " << chain_model->get_nodes().size() << " nodes\n";
    std::cout << "   Model has " << chain_model->get_edges().size() << " edges\n";
    std::cout << "   Input nodes: " << chain_model->get_input_nodes().size() << "\n";
    std::cout << "   Output nodes: " << chain_model->get_output_nodes().size() << "\n\n";
    
    // Display the model structure
    std::cout << "3. Model structure:\n";
    std::cout << "   Nodes: ";
    for (const auto& node : chain_model->get_nodes()) {
        std::cout << node->name() << " ";
    }
    std::cout << "\n   Edges: ";
    for (const auto& edge : chain_model->get_edges()) {
        std::cout << "(" << edge.first->name() << "->" << edge.second->name() << ") ";
    }
    std::cout << "\n\n";
    
    // Test the model with some data
    std::cout << "4. Testing model forward pass:\n";
    Matrix input_data = Matrix::Random(3, 2);  // 3 features, 2 samples (or could be time steps)
    std::cout << "   Input data shape: " << input_data.rows() << "x" << input_data.cols() << "\n";
    
    // Initialize and run the model
    chain_model->initialize(&input_data);
    std::cout << "   Model initialized successfully\n";
    
    Matrix output = chain_model->forward(input_data);
    std::cout << "   Output data shape: " << output.rows() << "x" << output.cols() << "\n";
    std::cout << "   Forward pass completed successfully\n\n";
    
    // Create a parallel branches model using & operator
    std::cout << "5. Creating a more complex model with parallel branches:\n";
    auto input_node2 = std::make_shared<Node>("input2");
    auto branch1 = std::make_shared<Node>("branch1");
    auto branch2 = std::make_shared<Node>("branch2");
    
    // Create separate branch models
    auto branch1_model = input_node2 >> branch1;
    auto branch2_model = input_node2 >> branch2;
    
    // Note: For proper parallel processing, we'd need more sophisticated merging logic
    std::cout << "   Branch 1 model created with " << branch1_model->get_nodes().size() << " nodes\n";
    std::cout << "   Branch 2 model created with " << branch2_model->get_nodes().size() << " nodes\n\n";
    
    // Test copying
    std::cout << "6. Testing model copy functionality:\n";
    auto copied_model = std::static_pointer_cast<Model>(chain_model->copy("copied_chain"));
    std::cout << "   Original model name: " << chain_model->name() << "\n";
    std::cout << "   Copied model name: " << copied_model->name() << "\n";
    std::cout << "   Copied model has " << copied_model->get_nodes().size() << " nodes\n";
    std::cout << "   Copy completed successfully\n\n";
    
    // Test error handling with cycles
    std::cout << "7. Testing cycle detection:\n";
    try {
        auto node_a = std::make_shared<Node>("A");
        auto node_b = std::make_shared<Node>("B");
        std::vector<NodePtr> cycle_nodes = {node_a, node_b};
        std::vector<Edge> cycle_edges = {{node_a, node_b}, {node_b, node_a}};  // Cycle: A->B->A
        Model cycle_model(cycle_nodes, cycle_edges);
        std::cout << "   ERROR: Cycle was not detected!\n";
    } catch (const std::runtime_error& e) {
        std::cout << "   Cycle correctly detected and rejected: " << e.what() << "\n";
    }
    
    std::cout << "\nModel example completed successfully!\n";
    std::cout << "The Model class provides a powerful way to compose complex\n";
    std::cout << "computational graphs from individual nodes using simple operators.\n";
    
    return 0;
}