/**
 * @file test_node.cpp
 * @brief Test basic node functionality
 */

#include <iostream>
#include "reservoircpp/types.hpp"
#include "reservoircpp/utils.hpp"

using namespace reservoircpp;

int main() {
    std::cout << "Testing utils..." << std::endl;
    
    try {
        // Test random generator
        auto& gen = reservoircpp::utils::RandomGenerator::instance();
        gen.set_seed(42);
        
        Float val = gen.uniform();
        std::cout << "Random value: " << val << std::endl;
        
        // Test matrix creation
        Matrix mat = reservoircpp::utils::random_uniform(3, 3, -1.0, 1.0);
        std::cout << "Random matrix:" << std::endl;
        std::cout << mat << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Utils test completed successfully!" << std::endl;
    return 0;
}
