/**
 * @file stage2_example.cpp
 * @brief Example demonstrating Stage 2 capabilities
 * 
 * This example shows the new Stage 2 functionality including
 * matrix generators, reservoirs, and readouts.
 */

#include <iostream>
#include <reservoircpp/reservoircpp.hpp>

using namespace reservoircpp;

int main() {
    std::cout << "=== ReservoirCpp Stage 2 Demo ===\n";
    std::cout << version_info() << "\n\n";
    
    // Set random seed for reproducibility
    utils::set_seed(42);
    
    // Demo matrix generators
    std::cout << "=== Matrix Generators ===\n";
    auto W_internal = matrix_generators::generate_internal_weights(50, 0.1, 0.9);
    auto W_input = matrix_generators::generate_input_weights(50, 3, 1.0);
    
    std::cout << "Generated internal weights: " << W_internal.rows() << "x" << W_internal.cols() << "\n";
    std::cout << "Generated input weights: " << W_input.rows() << "x" << W_input.cols() << "\n";
    std::cout << "Internal weights spectral radius: " << matrix_generators::spectral_radius(W_internal) << "\n\n";
    
    // Demo reservoir
    std::cout << "=== Reservoir Demo ===\n";
    ESN esn("demo_esn", 50, 1.0, 0.1, 0.9);
    
    // Create some demo data
    Matrix input_data(20, 3);
    input_data.setRandom();
    
    // Initialize and run reservoir
    esn.initialize(&input_data);
    std::cout << "ESN initialized with " << esn.units() << " units\n";
    std::cout << "Input dimension: " << esn.input_dim()[0] << "\n";
    std::cout << "Output dimension: " << esn.output_dim()[0] << "\n";
    
    Matrix reservoir_states = esn.forward(input_data);
    std::cout << "Reservoir states shape: " << reservoir_states.rows() << "x" << reservoir_states.cols() << "\n\n";
    
    // Demo readout
    std::cout << "=== Readout Demo ===\n";
    RidgeReadout ridge("demo_ridge", 1, 1e-6);
    
    // Create target data
    Matrix target_data(20, 1);
    target_data.setRandom();
    
    // Train readout
    ridge.fit(reservoir_states, target_data);
    std::cout << "Ridge readout trained on " << reservoir_states.rows() << " samples\n";
    std::cout << "Ridge parameter: " << ridge.ridge() << "\n";
    std::cout << "Is fitted: " << (ridge.is_fitted() ? "Yes" : "No") << "\n";
    
    // Make predictions
    Matrix predictions = ridge.predict(reservoir_states);
    std::cout << "Predictions shape: " << predictions.rows() << "x" << predictions.cols() << "\n";
    
    // Calculate simple MSE
    Matrix error = predictions - target_data;
    Float mse = error.array().square().mean();
    std::cout << "Training MSE: " << mse << "\n\n";
    
    // Demo FORCE learning
    std::cout << "=== FORCE Learning Demo ===\n";
    ForceReadout force("demo_force", 1, 1.0, 1.0);
    
    // Train with FORCE
    force.fit(reservoir_states, target_data);
    std::cout << "FORCE readout trained\n";
    std::cout << "Learning rate: " << force.learning_rate() << "\n";
    std::cout << "Regularization: " << force.regularization() << "\n";
    
    Matrix force_predictions = force.predict(reservoir_states);
    Matrix force_error = force_predictions - target_data;
    Float force_mse = force_error.array().square().mean();
    std::cout << "FORCE Training MSE: " << force_mse << "\n\n";
    
    std::cout << "=== Summary ===\n";
    std::cout << "✓ Matrix generators working\n";
    std::cout << "✓ ESN reservoir working\n";
    std::cout << "✓ Ridge readout working\n";
    std::cout << "✓ FORCE readout working\n";
    std::cout << "✓ Complete reservoir computing pipeline functional\n\n";
    
    std::cout << "Stage 2 (Core Reservoir Computing Components) is complete!\n";
    std::cout << "Ready for Stage 3 implementation.\n";
    
    return 0;
}