/**
 * @file debug_ip.cpp
 * @brief Debug IntrinsicPlasticity issue
 */

#include <iostream>
#include <exception>
#include "reservoircpp/reservoir.hpp"

using namespace reservoircpp;

int main() {
    std::cout << "Debugging IntrinsicPlasticity..." << std::endl;
    
    try {
        IntrinsicPlasticity ip("test_ip", 3, 1.0, 0.0, 1.0, 0.1, 1, "tanh");
        
        Matrix x(10, 1);
        x.setConstant(0.5);  // Constant input
        
        std::cout << "Initializing IP..." << std::endl;
        ip.initialize(&x);
        std::cout << "IP initialized" << std::endl;
        
        // Store initial values
        Matrix a_before = ip.a();
        Matrix b_before = ip.b();
        
        std::cout << "Initial a: " << std::endl << a_before << std::endl;
        std::cout << "Initial b: " << std::endl << b_before << std::endl;
        
        std::cout << "Running partial fit..." << std::endl;
        ip.partial_fit(x, 2);
        std::cout << "Partial fit completed" << std::endl;
        
        Matrix a_after = ip.a();
        Matrix b_after = ip.b();
        
        std::cout << "Final a: " << std::endl << a_after << std::endl;
        std::cout << "Final b: " << std::endl << b_after << std::endl;
        
        // Check for NaN values
        bool a_has_nan = false, b_has_nan = false;
        for (int i = 0; i < a_after.size(); i++) {
            if (std::isnan(a_after(i))) a_has_nan = true;
            if (std::isnan(b_after(i))) b_has_nan = true;
        }
        
        std::cout << "A has NaN: " << a_has_nan << std::endl;
        std::cout << "B has NaN: " << b_has_nan << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Debug completed successfully!" << std::endl;
    return 0;
}
