/**
 * @file stage3_stage4_example.cpp
 * @brief Example demonstrating Stage 3 (observables) and Stage 4 (datasets) functionality
 * 
 * This example shows the complete py2cpp migration for iteration 4,
 * demonstrating reservoir computing with observables and datasets.
 */

#include <iostream>
#include <iomanip>
#include "reservoircpp/reservoircpp.hpp"

using namespace reservoircpp;
using namespace reservoircpp::datasets;
using namespace reservoircpp::observables;

int main() {
    std::cout << "=== ReservoirCpp Stage 3 & 4 Demo ===" << std::endl;
    std::cout << "Demonstrating observables and datasets functionality" << std::endl << std::endl;
    
    try {
        // Stage 4: Generate a chaotic time series (Mackey-Glass)
        std::cout << "1. Generating Mackey-Glass time series..." << std::endl;
        Matrix mg_data = mackey_glass(1000, 17, 0.2, 0.1, 10.0, 1.0, 1.2, 100);
        std::cout << "   Generated " << mg_data.rows() << " samples" << std::endl;
        std::cout << "   Range: [" << mg_data.minCoeff() << ", " << mg_data.maxCoeff() << "]" << std::endl;
        
        // Prepare data for forecasting
        auto [X, y] = to_forecasting(mg_data, 1);
        auto [X_train, X_test, y_train, y_test] = to_forecasting_with_split(mg_data, 1, 200);
        
        std::cout << "   Training samples: " << X_train.rows() << std::endl;
        std::cout << "   Test samples: " << X_test.rows() << std::endl << std::endl;
        
        // Create a reservoir
        std::cout << "2. Creating Echo State Network..." << std::endl;
        Reservoir reservoir("esn", 100, 0.95, "tanh", 0.1, 0.95, 1.0, 0.1);
        reservoir.initialize(&X_train, &y_train);
        
        // Generate reservoir states
        Matrix states_train(X_train.rows(), 100);
        Matrix states_test(X_test.rows(), 100);
        
        reservoir.reset();
        for (int t = 0; t < X_train.rows(); ++t) {
            Matrix input = X_train.row(t);  // Keep as row vector
            Matrix state = reservoir.forward(input);
            states_train.row(t) = state;
        }
        
        reservoir.reset();
        for (int t = 0; t < X_test.rows(); ++t) {
            Matrix input = X_test.row(t);  // Keep as row vector
            Matrix state = reservoir.forward(input);
            states_test.row(t) = state;
        }
        
        std::cout << "   Reservoir states generated" << std::endl << std::endl;
        
        // Stage 3: Compute observables
        std::cout << "3. Computing reservoir observables..." << std::endl;
        
        // Spectral radius of reservoir weights
        Float spec_radius = spectral_radius(reservoir.W());
        std::cout << "   Spectral radius: " << std::fixed << std::setprecision(4) << spec_radius << std::endl;
        
        // Effective spectral radius from dynamics
        Float eff_spec_radius = effective_spectral_radius(states_train);
        std::cout << "   Effective spectral radius: " << std::fixed << std::setprecision(4) << eff_spec_radius << std::endl;
        
        // Memory capacity
        Float mem_cap = memory_capacity(states_train, X_train, 20);
        std::cout << "   Memory capacity (delay=20): " << std::fixed << std::setprecision(4) << mem_cap << std::endl << std::endl;
        
        // Train a readout
        std::cout << "4. Training Ridge readout..." << std::endl;
        RidgeReadout readout("ridge", 1, 1e-6);
        readout.fit(states_train, y_train);
        
        // Make predictions
        Matrix y_pred_test = readout.predict(states_test);
        
        // Stage 3: Evaluate performance with observables
        std::cout << "5. Evaluating prediction performance..." << std::endl;
        
        Float mse_score = mse(y_test, y_pred_test);
        Float rmse_score = rmse(y_test, y_pred_test);
        Float nrmse_score = nrmse(y_test, y_pred_test, "var");
        Float r2_score = rsquare(y_test, y_pred_test);
        
        std::cout << "   MSE: " << std::fixed << std::setprecision(6) << mse_score << std::endl;
        std::cout << "   RMSE: " << std::fixed << std::setprecision(6) << rmse_score << std::endl;
        std::cout << "   NRMSE: " << std::fixed << std::setprecision(4) << nrmse_score << std::endl;
        std::cout << "   R²: " << std::fixed << std::setprecision(4) << r2_score << std::endl << std::endl;
        
        // Stage 4: Try other datasets
        std::cout << "6. Testing other chaotic datasets..." << std::endl;
        
        // Lorenz system
        Matrix lorenz_data = lorenz(500, 0.01, 10.0, 28.0, 8.0/3.0);
        std::cout << "   Lorenz system: " << lorenz_data.rows() << " samples, 3D" << std::endl;
        
        // Hénon map
        Matrix henon_data = henon_map(300, 1.4, 0.3);
        std::cout << "   Hénon map: " << henon_data.rows() << " samples, 2D" << std::endl;
        
        // NARMA task
        auto [narma_input, narma_target] = narma(400, 10);
        std::cout << "   NARMA-10: " << narma_input.rows() << " samples" << std::endl;
        
        // MSO tasks
        Matrix mso2_data = mso2(200);
        Matrix mso8_data = mso8(200);
        std::cout << "   MSO-2: " << mso2_data.rows() << " samples" << std::endl;
        std::cout << "   MSO-8: " << mso8_data.rows() << " samples" << std::endl << std::endl;
        
        std::cout << "=== Demo completed successfully! ===" << std::endl;
        std::cout << "Stage 3 (observables) and Stage 4 (datasets) are fully functional." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}