/**
 * @file ops_example.cpp
 * @brief Example demonstrating the new ops functionality
 * 
 * This example shows how to use the new link, merge, and feedback operations
 * to create complex reservoir computing models using the C++ API.
 */

#include <iostream>
#include <reservoircpp/reservoircpp.hpp>

using namespace reservoircpp;

int main() {
    std::cout << "=== ReservoirCpp Ops Example ===" << std::endl;
    
    // Create some basic nodes
    auto input_node = std::make_shared<Node>("input");
    auto reservoir1 = std::make_shared<Node>("reservoir1");
    auto reservoir2 = std::make_shared<Node>("reservoir2");
    auto readout = std::make_shared<Node>("readout");
    
    std::cout << "Created nodes: " << input_node->name() << ", " 
              << reservoir1->name() << ", " << reservoir2->name() 
              << ", " << readout->name() << std::endl;
    
    // Example 1: Simple linking using ops::link
    std::cout << "\n1. Creating simple chain using ops::link..." << std::endl;
    auto simple_chain = ops::link(input_node, reservoir1, "simple_chain");
    std::cout << "Created model '" << simple_chain->name() 
              << "' with " << simple_chain->get_nodes().size() << " nodes" << std::endl;
    
    // Example 2: Link multiple nodes to one (many-to-one with concat)
    std::cout << "\n2. Creating many-to-one connection..." << std::endl;
    std::vector<NodePtr> multiple_inputs = {reservoir1, reservoir2};
    auto many_to_one = ops::link(multiple_inputs, readout, "many_to_one");
    std::cout << "Created model '" << many_to_one->name() 
              << "' with " << many_to_one->get_nodes().size() << " nodes (includes concat)" << std::endl;
    
    // Example 3: Link one to many (broadcast)
    std::cout << "\n3. Creating one-to-many connection..." << std::endl;
    std::vector<NodePtr> multiple_outputs = {reservoir1, reservoir2};
    auto one_to_many = ops::link(input_node, multiple_outputs, "one_to_many");
    std::cout << "Created model '" << one_to_many->name() 
              << "' with " << one_to_many->get_nodes().size() << " nodes" << std::endl;
    
    // Example 4: Merge models
    std::cout << "\n4. Merging models..." << std::endl;
    auto merged_model = ops::merge(simple_chain, many_to_one, "merged_system");
    std::cout << "Created merged model '" << merged_model->name() 
              << "' with " << merged_model->get_nodes().size() << " nodes" << std::endl;
    
    // Example 5: Using operator overloads (equivalent to ops functions)
    std::cout << "\n5. Using operator overloads..." << std::endl;
    auto chain_with_operators = input_node >> reservoir1 >> readout;
    std::cout << "Created chain using >> operator with " 
              << chain_with_operators->get_nodes().size() << " nodes" << std::endl;
    
    // Example 6: Feedback connections (basic)
    std::cout << "\n6. Creating feedback connection..." << std::endl;
    auto feedback_node = ops::link_feedback(reservoir1, reservoir2, false, "feedback_reservoir");
    std::cout << "Created feedback node '" << feedback_node->name() << "'" << std::endl;
    
    // Example 7: Complex model construction
    std::cout << "\n7. Building complex model..." << std::endl;
    
    // Create separate nodes for the complex model to avoid duplicates
    auto input_node2 = std::make_shared<Node>("input2");
    auto reservoir3 = std::make_shared<Node>("reservoir3");
    auto reservoir4 = std::make_shared<Node>("reservoir4");
    auto reservoir5 = std::make_shared<Node>("reservoir5");
    auto readout2 = std::make_shared<Node>("readout2");
    
    // Create input processing chain
    auto input_processing = input_node2 >> reservoir3;
    
    // Create parallel reservoir processing
    std::vector<NodePtr> parallel_outputs = {reservoir4, reservoir5}; // Use different nodes
    auto parallel_reservoirs = ops::link(reservoir3, parallel_outputs, "parallel");
    
    // Create readout with multiple inputs
    auto final_readout = ops::link(parallel_outputs, readout2, "final_readout");
    
    // Merge everything into one big model
    auto complex_model = ops::merge(input_processing, 
                                   ops::merge(parallel_reservoirs, final_readout, "processing"),
                                   "complex_esn");
    
    std::cout << "Created complex model '" << complex_model->name() 
              << "' with " << complex_model->get_nodes().size() << " nodes" << std::endl;
    
    // Print some model information
    std::cout << "\n=== Model Summary ===" << std::endl;
    std::cout << "Input nodes: " << complex_model->get_input_nodes().size() << std::endl;
    std::cout << "Output nodes: " << complex_model->get_output_nodes().size() << std::endl;
    std::cout << "Total edges: " << complex_model->get_edges().size() << std::endl;
    
    std::cout << "\nOps example completed successfully!" << std::endl;
    
    return 0;
}