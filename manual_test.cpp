/**
 * @file manual_test.cpp
 * @brief Manual test to check basic compilation
 */

#include <iostream>
#include <vector>

// Try to include basic headers to test compilation
#include "reservoircpp/types.hpp"
#include "reservoircpp/version.hpp"

using namespace reservoircpp;

int main() {
    std::cout << "Testing basic compilation..." << std::endl;
    
    // Test basic types
    Matrix m(2, 2);
    m << 1.0, 2.0, 3.0, 4.0;
    
    std::cout << "Matrix created successfully: " << std::endl;
    std::cout << m << std::endl;
    
    return 0;
}
