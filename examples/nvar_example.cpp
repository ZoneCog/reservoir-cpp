/**
 * @file nvar_example.cpp
 * @brief Example demonstrating NVAR (Nonlinear Vector Autoregressive) functionality
 * 
 * This example shows how to use the NVAR node for feature expansion
 * combining delayed inputs with nonlinear monomial transformations.
 */

#include <iostream>
#include <iomanip>
#include <reservoircpp/reservoircpp.hpp>

using namespace reservoircpp;

int main() {
    std::cout << "=== ReservoirCpp NVAR Demo ===\n";
    std::cout << version_info() << "\n\n";
    
    // Set random seed for reproducibility
    utils::set_seed(42);
    
    std::cout << "=== Creating NVAR Node ===\n";
    NVAR nvar(
        "nvar_demo",       // name
        3,                 // delay (use last 3 timesteps)
        2,                 // order (quadratic monomials)
        1                  // strides (every timestep)
    );
    
    std::cout << "Created NVAR with:\n";
    std::cout << "  - Delay: " << nvar.delay() << " timesteps\n";
    std::cout << "  - Order: " << nvar.order() << " (quadratic monomials)\n";
    std::cout << "  - Strides: " << nvar.strides() << "\n\n";
    
    // Create simple test data
    std::cout << "=== Generating Test Data ===\n";
    int seq_length = 10;
    int input_dim = 2;
    
    Matrix test_data(seq_length, input_dim);
    
    // Generate a simple sequence for demonstration
    for (int t = 0; t < seq_length; ++t) {
        test_data(t, 0) = static_cast<Float>(t + 1);      // 1, 2, 3, 4, ...
        test_data(t, 1) = static_cast<Float>((t + 1) * 2); // 2, 4, 6, 8, ...
    }
    
    std::cout << "Input sequence:\n";
    for (int t = 0; t < seq_length; ++t) {
        std::cout << "  t=" << t << ": [" << test_data(t, 0) << ", " << test_data(t, 1) << "]\n";
    }
    std::cout << "\n";
    
    // Initialize NVAR
    std::cout << "=== Initializing NVAR ===\n";
    nvar.initialize(&test_data);
    
    std::cout << "NVAR dimensions after initialization:\n";
    std::cout << "  - Input dimension: " << nvar.input_dim()[0] << "\n";
    std::cout << "  - Linear features: " << nvar.linear_dim() << " (delay × input_dim = " 
              << nvar.delay() << " × " << nvar.input_dim()[0] << ")\n";
    std::cout << "  - Nonlinear features: " << nvar.nonlinear_dim() 
              << " (monomials of order " << nvar.order() << ")\n";
    std::cout << "  - Total output dimension: " << nvar.output_dim()[0] << "\n\n";
    
    // Process the sequence
    std::cout << "=== Forward Pass ===\n";
    Matrix nvar_output = nvar.forward(test_data);
    
    std::cout << "NVAR output shape: " << nvar_output.rows() << " × " << nvar_output.cols() << "\n\n";
    
    // Show detailed output for first few timesteps
    std::cout << "=== Detailed Output Analysis ===\n";
    for (int t = 0; t < std::min(5, seq_length); ++t) {
        std::cout << "Timestep " << t << ":\n";
        std::cout << "  Input: [" << test_data(t, 0) << ", " << test_data(t, 1) << "]\n";
        
        // Show linear features (delayed inputs)
        std::cout << "  Linear features: [";
        for (int i = 0; i < nvar.linear_dim(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(1) << nvar_output(t, i);
        }
        std::cout << "]\n";
        
        // Show nonlinear features
        std::cout << "  Nonlinear features: [";
        for (int i = 0; i < nvar.nonlinear_dim(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(1) << nvar_output(t, nvar.linear_dim() + i);
        }
        std::cout << "]\n\n";
    }
    
    // Demonstrate the feature expansion
    std::cout << "=== Feature Expansion Demonstration ===\n";
    std::cout << "At timestep 3 (input [4.0, 8.0]):\n";
    if (seq_length > 3) {
        int t = 3;
        std::cout << "Linear features represent:\n";
        std::cout << "  - Current input: [" << nvar_output(t, 0) << ", " << nvar_output(t, 1) << "]\n";
        std::cout << "  - t-1 input: [" << nvar_output(t, 2) << ", " << nvar_output(t, 3) << "]\n";
        std::cout << "  - t-2 input: [" << nvar_output(t, 4) << ", " << nvar_output(t, 5) << "]\n";
        std::cout << "\nNonlinear features are products of linear features:\n";
        std::cout << "  - Example monomials: x₁², x₁×x₂, x₂², x₁×x₃, ...\n";
    }
    std::cout << "\n";
    
    // Demonstrate with different parameters
    std::cout << "=== Different NVAR Configurations ===\n";
    
    // Higher order
    NVAR nvar_order3("nvar_order3", 2, 3, 1);  // Order 3 (cubic)
    nvar_order3.initialize(&test_data);
    std::cout << "NVAR with order=3:\n";
    std::cout << "  - Linear features: " << nvar_order3.linear_dim() << "\n";
    std::cout << "  - Nonlinear features: " << nvar_order3.nonlinear_dim() << "\n";
    std::cout << "  - Total output: " << nvar_order3.output_dim()[0] << "\n\n";
    
    // With strides
    NVAR nvar_strides("nvar_strides", 4, 2, 2);  // Every 2nd timestep
    nvar_strides.initialize(&test_data);
    std::cout << "NVAR with strides=2:\n";
    std::cout << "  - Delay: " << nvar_strides.delay() << ", Strides: " << nvar_strides.strides() << "\n";
    std::cout << "  - Linear features: " << nvar_strides.linear_dim() << "\n";
    std::cout << "  - Nonlinear features: " << nvar_strides.nonlinear_dim() << "\n";
    std::cout << "  - Total output: " << nvar_strides.output_dim()[0] << "\n\n";
    
    // Practical application example
    std::cout << "=== Practical Application ===\n";
    std::cout << "NVAR can be used for:\n";
    std::cout << "• Time series prediction with nonlinear relationships\n";
    std::cout << "• Feature engineering for chaotic systems (Lorenz, etc.)\n";
    std::cout << "• Next Generation Reservoir Computing architectures\n";
    std::cout << "• Replacing traditional reservoirs in some applications\n\n";
    
    // Show copy functionality
    std::cout << "=== Copy Functionality ===\n";
    auto nvar_copy = nvar.copy("nvar_copy");
    auto copied_nvar = dynamic_cast<NVAR*>(nvar_copy.get());
    
    std::cout << "Copied NVAR:\n";
    std::cout << "  - Name: " << copied_nvar->name() << "\n";
    std::cout << "  - Same dimensions: " << (copied_nvar->output_dim()[0] == nvar.output_dim()[0] ? "✓" : "✗") << "\n";
    std::cout << "  - Same parameters: " << 
                 (copied_nvar->delay() == nvar.delay() && 
                  copied_nvar->order() == nvar.order() && 
                  copied_nvar->strides() == nvar.strides() ? "✓" : "✗") << "\n\n";
    
    std::cout << "=== Demo Complete ===\n";
    std::cout << "NVAR successfully creates rich feature representations\n";
    std::cout << "by combining delayed inputs with nonlinear transformations!\n";
    
    return 0;
}