/**
 * @file simple_example.cpp
 * @brief Simple example demonstrating the current Stage 0 capabilities
 * 
 * This example shows basic usage of the ReservoirCpp types and demonstrates
 * that the build system and dependencies are working correctly.
 */

#include <iostream>
#include <reservoircpp/reservoircpp.hpp>

int main() {
    std::cout << "=== ReservoirCpp Stage 0 Demo ===" << std::endl;
    std::cout << reservoircpp::version_info() << std::endl;
    std::cout << std::endl;
    
    // Demonstrate basic matrix operations using Eigen
    std::cout << "Creating a 3x3 matrix:" << std::endl;
    reservoircpp::Matrix m(3, 3);
    m << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    
    std::cout << m << std::endl;
    std::cout << std::endl;
    
    // Demonstrate vector operations
    std::cout << "Creating a vector:" << std::endl;
    reservoircpp::Vector v(3);
    v << 1.0, 2.0, 3.0;
    
    std::cout << v.transpose() << std::endl;
    std::cout << std::endl;
    
    // Demonstrate matrix-vector multiplication
    std::cout << "Matrix-vector multiplication:" << std::endl;
    reservoircpp::Vector result = m * v;
    std::cout << result.transpose() << std::endl;
    std::cout << std::endl;
    
    // Demonstrate shape functionality
    std::cout << "Shape operations:" << std::endl;
    reservoircpp::Shape shape = {100, 50};  // 100 units, 50 inputs
    std::cout << "Reservoir size: " << shape[0] << " units, " 
              << shape[1] << " inputs" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Stage 0 infrastructure is working correctly!" << std::endl;
    std::cout << "Ready for Stage 1 implementation." << std::endl;
    
    return 0;
}