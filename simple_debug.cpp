/**
 * @file simple_debug.cpp
 * @brief Simple debugging of matrix operations
 */

#include <iostream>
#include <exception>
#include <cmath>
#include "reservoircpp/types.hpp"

using namespace reservoircpp;

int main() {
    std::cout << "Testing matrix operations..." << std::endl;
    
    try {
        Matrix a(3, 1);
        a.setOnes();
        
        std::cout << "a: " << a.transpose() << std::endl;
        
        Matrix a_inv = a.cwiseInverse();
        std::cout << "a_inv: " << a_inv.transpose() << std::endl;
        
        // Check if any values are infinite or NaN
        for (int i = 0; i < a_inv.size(); i++) {
            if (std::isinf(a_inv(i))) {
                std::cout << "Found infinity at index " << i << std::endl;
            }
            if (std::isnan(a_inv(i))) {
                std::cout << "Found NaN at index " << i << std::endl;
            }
        }
        
        std::cout << "Test completed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
