/**
 * @file practical_tutorial.cpp
 * @brief Practical ReservoirCpp Tutorial - Stage 6 Implementation
 * 
 * This practical tutorial demonstrates the essential reservoir computing workflow
 * using ReservoirCpp, showcasing core functionality and best practices.
 * 
 * Tutorial covers:
 * 1. Basic data generation
 * 2. Reservoir configuration
 * 3. Training with different readouts
 * 4. Performance evaluation
 * 5. Essential features overview
 */

#include <iostream>
#include <iomanip>
#include <reservoircpp/reservoircpp.hpp>

using namespace reservoircpp;

void print_section_header(const std::string& title) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

void print_subsection(const std::string& title) {
    std::cout << "\n--- " << title << " ---" << std::endl;
}

int main() {
    std::cout << "ReservoirCpp Practical Tutorial" << std::endl;
    std::cout << "Essential Reservoir Computing in C++" << std::endl;
    std::cout << version_info() << std::endl;
    
    // Set reproducible seed
    utils::set_seed(42);
    
    print_section_header("1. DATA GENERATION");
    
    // Generate simple datasets
    print_subsection("Mackey-Glass Time Series");
    Matrix mg_data = datasets::mackey_glass(500);
    std::cout << "Generated " << mg_data.rows() << " time steps" << std::endl;
    std::cout << "Sample values: " << mg_data.block(0, 0, 5, 1).transpose() << std::endl;
    
    print_subsection("Lorenz Attractor");
    Matrix lorenz_data = datasets::lorenz(300);
    std::cout << "Generated Lorenz data: " << lorenz_data.rows() << "x" << lorenz_data.cols() << std::endl;
    std::cout << "First 3 points: " << std::endl << lorenz_data.block(0, 0, 3, 3) << std::endl;
    
    print_subsection("Data Preparation");
    // Create forecasting task from Mackey-Glass
    auto [X_train, y_train] = datasets::to_forecasting(mg_data, 1);
    
    std::cout << "Forecasting data shape: " << X_train.rows() << "x" << X_train.cols() << std::endl;
    
    // Use most of the data for training, small portion for testing
    Matrix X_tr, y_tr, X_test, y_test;
    if (X_train.cols() > 10) {
        int train_size = X_train.cols() - 10;  // Reserve last 10 for testing
        X_tr = X_train.leftCols(train_size);
        y_tr = y_train.leftCols(train_size);
        X_test = X_train.rightCols(10);
        y_test = y_train.rightCols(10);
        
        std::cout << "Training samples: " << X_tr.cols() << std::endl;
        std::cout << "Test samples: " << X_test.cols() << std::endl;
    } else {
        // If too small, use all for training
        X_tr = X_train;
        y_tr = y_train;
        X_test = X_train.rightCols(1);  // Just use last sample for testing
        y_test = y_train.rightCols(1);
        std::cout << "Dataset too small, using all for training" << std::endl;
    }
    
    print_section_header("2. RESERVOIR CONFIGURATION");
    
    print_subsection("Basic Reservoir");
    auto reservoir = std::make_unique<Reservoir>("basic_reservoir", 100);
    
    std::cout << "Reservoir created with default parameters" << std::endl;
    std::cout << "  Units: 100" << std::endl;
    std::cout << "  Spectral radius: 0.9 (default)" << std::endl;
    std::cout << "  Input scaling: 1.0 (default)" << std::endl;
    
    print_subsection("Echo State Network");
    auto esn = std::make_unique<ESN>("esn", 80);
    
    std::cout << "ESN created with 80 units" << std::endl;
    
    print_section_header("3. TRAINING AND PREDICTION");
    
    print_subsection("Reservoir Initialization");
    reservoir->initialize(&X_tr);
    std::cout << "Reservoir initialized" << std::endl;
    
    print_subsection("State Generation");
    auto train_states = reservoir->forward(X_tr);
    std::cout << "Training states shape: " << train_states.rows() << "x" << train_states.cols() << std::endl;
    
    print_subsection("Ridge Regression Training");
    auto ridge = std::make_unique<RidgeReadout>("ridge", y_tr.cols());  // Use cols() for number of outputs
    
    ridge->fit(train_states.transpose(), y_tr.transpose());  // Transpose to match expected format
    std::cout << "Ridge readout trained" << std::endl;
    
    // Make predictions
    auto y_pred_train = ridge->forward(train_states.transpose());
    
    reservoir->reset();
    auto test_states = reservoir->forward(X_test);
    auto y_pred_test = ridge->forward(test_states.transpose());
    
    print_section_header("4. PERFORMANCE EVALUATION");
    
    std::cout << std::fixed << std::setprecision(6);
    
    print_subsection("Training Performance");
    Float mse_train = observables::mse(y_tr.transpose(), y_pred_train);
    Float rmse_train = observables::rmse(y_tr.transpose(), y_pred_train);
    Float r2_train = observables::rsquare(y_tr.transpose(), y_pred_train);
    
    std::cout << "MSE:  " << mse_train << std::endl;
    std::cout << "RMSE: " << rmse_train << std::endl;
    std::cout << "R²:   " << r2_train << std::endl;
    
    print_subsection("Test Performance");
    Float mse_test = observables::mse(y_test.transpose(), y_pred_test);
    Float rmse_test = observables::rmse(y_test.transpose(), y_pred_test);
    Float r2_test = observables::rsquare(y_test.transpose(), y_pred_test);
    
    std::cout << "MSE:  " << mse_test << std::endl;
    std::cout << "RMSE: " << rmse_test << std::endl;
    std::cout << "R²:   " << r2_test << std::endl;
    
    print_subsection("Reservoir Analysis");
    Matrix W = matrix_generators::generate_internal_weights(50, 0.2f, 0.9f);
    Float sr = observables::spectral_radius(W);
    std::cout << "Sample reservoir spectral radius: " << sr << std::endl;
    
    print_section_header("5. ACTIVATION FUNCTIONS");
    
    print_subsection("Available Activations");
    Matrix test_input(1, 5);
    test_input << -2.0, -1.0, 0.0, 1.0, 2.0;
    
    std::cout << "Input: " << test_input << std::endl;
    std::cout << "Sigmoid: " << activations::sigmoid(test_input) << std::endl;
    std::cout << "Tanh: " << activations::tanh(test_input) << std::endl;
    std::cout << "ReLU: " << activations::relu(test_input) << std::endl;
    
    print_subsection("Activation Registry");
    auto sigmoid_fn = activations::get_function("sigmoid");
    auto relu_fn = activations::get_function("relu");
    
    std::cout << "Registry sigmoid: " << sigmoid_fn(test_input) << std::endl;
    std::cout << "Registry ReLU: " << relu_fn(test_input) << std::endl;
    
    print_section_header("6. EXPERIMENTAL FEATURES");
    
    print_subsection("LIF Spiking Neuron");
    experimental::LIF lif("spiking", 10);
    Matrix spike_input = Matrix::Constant(1, 10, 1.5f);
    auto spikes = lif.forward(spike_input);
    std::cout << "Spike count: " << spikes.sum() << std::endl;
    
    print_subsection("Add Node");
    experimental::Add add_node("add");
    Matrix a = Matrix::Random(2, 3);
    Matrix b = Matrix::Random(2, 3);
    auto sum_result = add_node.forward(a, b);
    std::cout << "Matrix addition completed" << std::endl;
    
    print_section_header("7. DATASETS AND UTILITIES");
    
    print_subsection("Available Datasets");
    std::cout << "✓ Mackey-Glass time series" << std::endl;
    std::cout << "✓ Lorenz chaotic attractor" << std::endl;
    std::cout << "✓ Hénon map" << std::endl;
    std::cout << "✓ Logistic map" << std::endl;
    std::cout << "✓ NARMA task" << std::endl;
    std::cout << "✓ MSO (Multiple Superimposed Oscillators)" << std::endl;
    
    print_subsection("Matrix Generators");
    Matrix uniform_mat = matrix_generators::uniform(3, 3, -1.0, 1.0);
    Matrix normal_mat = matrix_generators::normal(3, 3, 0.0, 1.0);
    
    std::cout << "Uniform matrix:\n" << uniform_mat << std::endl;
    std::cout << "\nNormal matrix:\n" << normal_mat << std::endl;
    
    print_section_header("8. MODEL SERIALIZATION");
    
    print_subsection("Export Model Configuration");
    bool exported = compat::ModelSerializer::export_to_python(*reservoir, "/tmp/tutorial_model.json");
    std::cout << "Model export: " << (exported ? "Success" : "Failed") << std::endl;
    
    print_section_header("SUMMARY");
    
    std::cout << "\n✅ TUTORIAL COMPLETED SUCCESSFULLY" << std::endl;
    std::cout << "\nKey features demonstrated:" << std::endl;
    std::cout << "• Data generation and preprocessing" << std::endl;
    std::cout << "• Reservoir and ESN configuration" << std::endl;
    std::cout << "• Ridge regression training" << std::endl;
    std::cout << "• Performance evaluation metrics" << std::endl;
    std::cout << "• Activation functions and registry" << std::endl;
    std::cout << "• Experimental nodes (LIF, Add)" << std::endl;
    std::cout << "• Matrix generators and utilities" << std::endl;
    std::cout << "• Model serialization" << std::endl;
    
    std::cout << "\nNext steps:" << std::endl;
    std::cout << "• Try different activation functions" << std::endl;
    std::cout << "• Experiment with hyperparameters" << std::endl;
    std::cout << "• Use different datasets" << std::endl;
    std::cout << "• Compare FORCE vs Ridge readouts" << std::endl;
    std::cout << "• Explore advanced features" << std::endl;
    
    std::cout << "\nFor more examples, see examples/ directory." << std::endl;
    std::cout << "Documentation available in docs/ directory." << std::endl;
    
    return 0;
}