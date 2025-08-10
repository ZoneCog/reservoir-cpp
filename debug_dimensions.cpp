/**
 * @file debug_dimensions.cpp
 * @brief Debug matrix dimensions in IP
 */

#include <iostream>
#include <exception>
#include "reservoircpp/reservoir.hpp"

using namespace reservoircpp;

int main() {
    std::cout << "Debugging matrix dimensions..." << std::endl;
    
    try {
        // Create a small IP with explicit dimensions
        IntrinsicPlasticity ip("test_ip", 3, 1.0, 0.0, 1.0, 0.01, 1, "tanh");
        
        Matrix x(5, 2);  // 5 timesteps, 2 features
        x.setConstant(0.5);
        
        std::cout << "Input shape: " << x.rows() << "x" << x.cols() << std::endl;
        
        ip.initialize(&x);
        std::cout << "IP initialized" << std::endl;
        
        // Check initial IP parameter dimensions
        Matrix a = ip.a();
        Matrix b = ip.b();
        
        std::cout << "IP parameters a shape: " << a.rows() << "x" << a.cols() << std::endl;
        std::cout << "IP parameters b shape: " << b.rows() << "x" << b.cols() << std::endl;
        std::cout << "Units: " << ip.units() << std::endl;
        
        // Check initial internal_state dimensions (through forward pass)
        ip.forward(x);
        
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Debug completed successfully!" << std::endl;
    return 0;
}
