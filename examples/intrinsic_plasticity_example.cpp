/**
 * @file intrinsic_plasticity_example.cpp
 * @brief Example demonstrating IntrinsicPlasticity functionality
 * 
 * This example shows how to use the IntrinsicPlasticity reservoir
 * for learning neuron intrinsic properties to match target distributions.
 */

#include <iostream>
#include <iomanip>
#include <reservoircpp/reservoircpp.hpp>

using namespace reservoircpp;

int main() {
    std::cout << "=== ReservoirCpp IntrinsicPlasticity Demo ===\n";
    std::cout << version_info() << "\n\n";
    
    // Set random seed for reproducibility
    utils::set_seed(42);
    
    // Create IntrinsicPlasticity reservoir with tanh activation
    std::cout << "=== Creating IntrinsicPlasticity Reservoir ===\n";
    IntrinsicPlasticity ip_reservoir(
        "ip_demo",         // name
        50,                // units
        1.0,               // leak rate
        0.0,               // target mean (mu)
        0.1,               // target variance (sigma)  
        5e-3,              // learning rate
        5,                 // epochs
        "tanh"             // activation function
    );
    
    std::cout << "Created IP reservoir with:\n";
    std::cout << "  - Units: " << ip_reservoir.units() << "\n";
    std::cout << "  - Target mean (μ): " << ip_reservoir.mu() << "\n";
    std::cout << "  - Target variance (σ): " << ip_reservoir.sigma() << "\n";
    std::cout << "  - Learning rate: " << ip_reservoir.learning_rate() << "\n";
    std::cout << "  - Epochs: " << ip_reservoir.epochs() << "\n";
    std::cout << "  - Activation: " << ip_reservoir.activation_name() << "\n\n";
    
    // Generate training data (NARMA-like sequence)
    std::cout << "=== Generating Training Data ===\n";
    int seq_length = 100;
    int input_dim = 2;
    
    Matrix training_data(seq_length, input_dim);
    
    // Generate a simple nonlinear sequence
    for (int t = 0; t < seq_length; ++t) {
        training_data(t, 0) = 0.3 * std::sin(0.1 * t) + 0.7 * std::cos(0.05 * t);
        training_data(t, 1) = 0.4 * std::sin(0.08 * t + 1.0) + 0.6 * std::cos(0.12 * t + 0.5);
    }
    
    std::cout << "Generated training sequence of length " << seq_length 
              << " with " << input_dim << " input dimensions\n\n";
    
    // Initialize reservoir
    std::cout << "=== Initializing Reservoir ===\n";
    ip_reservoir.initialize(&training_data);
    
    std::cout << "Initial IP parameters:\n";
    std::cout << "  - a (gain) mean: " << ip_reservoir.a().mean() << "\n";
    std::cout << "  - a (gain) std: " << std::sqrt((ip_reservoir.a().array() - ip_reservoir.a().mean()).square().mean()) << "\n";
    std::cout << "  - b (bias) mean: " << ip_reservoir.b().mean() << "\n";
    std::cout << "  - b (bias) std: " << std::sqrt((ip_reservoir.b().array() - ip_reservoir.b().mean()).square().mean()) << "\n\n";
    
    // Test forward pass before training
    std::cout << "=== Forward Pass Before Training ===\n";
    Matrix states_before = ip_reservoir.forward(training_data.topRows(10));
    
    Float mean_activation_before = states_before.mean();
    Float std_activation_before = std::sqrt((states_before.array() - mean_activation_before).square().mean());
    
    std::cout << "Activation statistics before training:\n";
    std::cout << "  - Mean: " << std::fixed << std::setprecision(4) << mean_activation_before << "\n";
    std::cout << "  - Std: " << std::fixed << std::setprecision(4) << std_activation_before << "\n\n";
    
    // Train the IP parameters
    std::cout << "=== Training IP Parameters ===\n";
    std::vector<Matrix> training_sequences = {training_data};
    
    std::cout << "Training with " << ip_reservoir.epochs() << " epochs...\n";
    ip_reservoir.fit(training_sequences, 10);  // 10 warmup steps
    
    std::cout << "Training completed!\n\n";
    
    // Show updated IP parameters
    std::cout << "=== Updated IP Parameters ===\n";
    std::cout << "Final IP parameters:\n";
    std::cout << "  - a (gain) mean: " << ip_reservoir.a().mean() << "\n";
    std::cout << "  - a (gain) std: " << std::sqrt((ip_reservoir.a().array() - ip_reservoir.a().mean()).square().mean()) << "\n";
    std::cout << "  - b (bias) mean: " << ip_reservoir.b().mean() << "\n";
    std::cout << "  - b (bias) std: " << std::sqrt((ip_reservoir.b().array() - ip_reservoir.b().mean()).square().mean()) << "\n\n";
    
    // Test forward pass after training
    std::cout << "=== Forward Pass After Training ===\n";
    ip_reservoir.reset();  // Reset state for fair comparison
    Matrix states_after = ip_reservoir.forward(training_data.topRows(10));
    
    Float mean_activation_after = states_after.mean();
    Float std_activation_after = std::sqrt((states_after.array() - mean_activation_after).square().mean());
    
    std::cout << "Activation statistics after training:\n";
    std::cout << "  - Mean: " << std::fixed << std::setprecision(4) << mean_activation_after << "\n";
    std::cout << "  - Std: " << std::fixed << std::setprecision(4) << std_activation_after << "\n";
    std::cout << "  - Target mean: " << ip_reservoir.mu() << "\n";
    std::cout << "  - Target std: " << ip_reservoir.sigma() << "\n\n";
    
    // Show improvement
    Float mean_error_before = std::abs(mean_activation_before - ip_reservoir.mu());
    Float mean_error_after = std::abs(mean_activation_after - ip_reservoir.mu());
    Float std_error_before = std::abs(std_activation_before - ip_reservoir.sigma());
    Float std_error_after = std::abs(std_activation_after - ip_reservoir.sigma());
    
    std::cout << "=== Training Results ===\n";
    std::cout << "Mean error improvement: " << std::fixed << std::setprecision(4) 
              << mean_error_before << " → " << mean_error_after;
    if (mean_error_after < mean_error_before) {
        std::cout << " (✓ improved by " << (mean_error_before - mean_error_after) << ")";
    }
    std::cout << "\n";
    
    std::cout << "Std error improvement: " << std::fixed << std::setprecision(4) 
              << std_error_before << " → " << std_error_after;
    if (std_error_after < std_error_before) {
        std::cout << " (✓ improved by " << (std_error_before - std_error_after) << ")";
    }
    std::cout << "\n\n";
    
    // Demonstrate with sigmoid activation
    std::cout << "=== Sigmoid Activation Demo ===\n";
    IntrinsicPlasticity ip_sigmoid(
        "ip_sigmoid",      // name
        30,                // units
        1.0,               // leak rate
        0.5,               // target mean (mu) for exponential distribution
        1.0,               // sigma (not used for sigmoid)
        1e-2,              // learning rate
        3,                 // epochs
        "sigmoid"          // activation function
    );
    
    ip_sigmoid.initialize(&training_data);
    std::cout << "Created sigmoid IP reservoir with target mean μ = " << ip_sigmoid.mu() << "\n";
    
    // Quick training
    std::vector<Matrix> short_sequences = {training_data.topRows(50)};
    ip_sigmoid.fit(short_sequences, 5);
    
    Matrix sigmoid_states = ip_sigmoid.forward(training_data.topRows(10));
    Float sigmoid_mean = sigmoid_states.mean();
    
    std::cout << "Sigmoid activation mean after training: " << std::fixed << std::setprecision(4) 
              << sigmoid_mean << " (target: " << ip_sigmoid.mu() << ")\n";
    std::cout << "All sigmoid outputs in [0,1]: " << 
                 (sigmoid_states.minCoeff() >= 0.0 && sigmoid_states.maxCoeff() <= 1.0 ? "✓" : "✗") << "\n\n";
    
    std::cout << "=== Demo Complete ===\n";
    std::cout << "IntrinsicPlasticity successfully adapts neuron parameters to achieve target distributions!\n";
    
    return 0;
}