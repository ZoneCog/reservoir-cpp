/**
 * @file test_reservoir.cpp
 * @brief Test reservoir functionality
 */

#include <iostream>
#include "reservoircpp/types.hpp"
#include "reservoircpp/reservoir.hpp"

using namespace reservoircpp;

int main() {
    std::cout << "Testing reservoir..." << std::endl;
    
    try {
        // Create a small reservoir
        Reservoir reservoir("test", 10, 0.5, "tanh", 0.1, 0.9, 1.0, 0.0);
        std::cout << "Reservoir created with " << reservoir.units() << " units" << std::endl;
        
        // Create input data
        Matrix input(1, 3); // 1D input, 3 timesteps
        input << 0.1, 0.5, 0.8;
        std::cout << "Input data:" << std::endl;
        std::cout << input << std::endl;
        
        // Initialize reservoir
        reservoir.initialize(&input);
        std::cout << "Reservoir initialized" << std::endl;
        
        // Forward pass
        Matrix output = reservoir.forward(input);
        std::cout << "Output shape: " << output.rows() << "x" << output.cols() << std::endl;
        std::cout << "Output data:" << std::endl;
        std::cout << output << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Reservoir test completed successfully!" << std::endl;
    return 0;
}
