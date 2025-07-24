/**
 * @file stage5_example.cpp
 * @brief Example demonstrating Stage 5 - Ancillary and Advanced Features
 */

#include <iostream>
#include <reservoircpp/reservoircpp.hpp>

using namespace reservoircpp;

int main() {
    std::cout << "=== ReservoirCpp Stage 5 - Ancillary and Advanced Features Example ===\n\n";

    // 1. Experimental Features
    std::cout << "1. Experimental Features\n";
    std::cout << "------------------------\n";
    
    // LIF Spiking Neuron
    std::cout << "Creating LIF spiking neuron with 5 units...\n";
    experimental::LIF lif("demo_lif", 5, 10.0f, 2.0f, 0.8f, 0.0f, 1.0f);
    
    Matrix input = Matrix::Constant(1, 5, 1.5);  // Strong input
    std::cout << "Input: " << input << "\n";
    
    Matrix output = lif.forward(input);
    std::cout << "LIF output (first step): " << output << "\n";
    
    // Run a few more steps to see spiking behavior
    for (int i = 0; i < 3; ++i) {
        output = lif.forward(input);
    }
    std::cout << "LIF output (after buildup): " << output << "\n\n";
    
    // Add Node
    std::cout << "Demonstrating Add node...\n";
    experimental::Add add_node("demo_add");
    
    Matrix a(1, 3);
    a << 1, 2, 3;
    Matrix b(1, 3);
    b << 10, 20, 30;
    
    Matrix sum = add_node.forward(a, b);
    std::cout << "a: " << a << "\n";
    std::cout << "b: " << b << "\n";
    std::cout << "a + b: " << sum << "\n\n";
    
    // RandomChoice
    std::cout << "Demonstrating RandomChoice feature selection...\n";
    experimental::RandomChoice choice("demo_choice", 3, 42);
    
    Matrix features(2, 8);
    for (int i = 0; i < 8; ++i) {
        features(0, i) = i;
        features(1, i) = i * i;
    }
    std::cout << "Original features shape: " << features.rows() << "x" << features.cols() << "\n";
    
    Matrix selected = choice.forward(features);
    std::cout << "Selected features shape: " << selected.rows() << "x" << selected.cols() << "\n";
    std::cout << "Selected features:\n" << selected << "\n\n";
    
    // 2. Hyperparameter Optimization
    std::cout << "2. Hyperparameter Optimization\n";
    std::cout << "------------------------------\n";
    
    // Define search space
    std::vector<hyper::ParameterSpace> search_space = {
        hyper::ParameterSpace::uniform("learning_rate", 0.001f, 0.1f),
        hyper::ParameterSpace::choice("units", {50, 100, 200}),
        hyper::ParameterSpace::log_uniform("regularization", 1e-6f, 1e-2f)
    };
    
    std::cout << "Created search space with " << search_space.size() << " parameters\n";
    
    // Create optimizer
    hyper::RandomSearch optimizer(search_space, 42);
    
    // Simple objective function (minimize distance from target)
    hyper::ObjectiveFunction objective = [](const hyper::HyperConfig& params) -> float {
        float lr = params.at("learning_rate");
        float units = params.at("units");
        float reg = params.at("regularization");
        
        // Dummy objective: prefer lr=0.01, units=100, reg=1e-4
        float lr_diff = std::abs(lr - 0.01f);
        float units_diff = std::abs(units - 100.0f) / 100.0f;
        float reg_diff = std::abs(std::log10(reg) - (-4.0f));
        
        return -(lr_diff + units_diff + reg_diff);  // Negative because we maximize
    };
    
    std::cout << "Running optimization with 20 trials...\n";
    hyper::OptimizationResult result = optimizer.optimize(objective, 20);
    
    std::cout << "Best score: " << result.best_score << "\n";
    std::cout << "Best parameters:\n";
    for (const auto& [name, value] : result.best_params) {
        std::cout << "  " << name << ": " << value << "\n";
    }
    std::cout << "Optimization took: " << result.optimization_time << " seconds\n\n";
    
    // 3. Plotting
    std::cout << "3. Plotting (Python Export)\n";
    std::cout << "----------------------------\n";
    
    // Generate some data
    Vector x = Vector::LinSpaced(10, 0, 9);
    Vector y = x.array().square();
    
    std::cout << "Creating plot data...\n";
    std::cout << "x: " << x.transpose() << "\n";
    std::cout << "y: " << y.transpose() << "\n";
    
    // Create plotter with Python export backend
    plotting::PlotConfig config;
    config.title = "Quadratic Function";
    config.xlabel = "x";
    config.ylabel = "y = x²";
    config.color = "blue";
    
    auto& plotter = plotting::PlotUtils::get_default_plotter();
    plotter.plot(x, y, config, "quadratic");
    plotter.save("/tmp/stage5_plot.png");
    
    std::cout << "Plot saved to /tmp/stage5_plot.png\n";
    std::cout << "Python plotting script generated in plots/ directory\n\n";
    
    // 4. Compatibility
    std::cout << "4. Compatibility\n";
    std::cout << "----------------\n";
    
    // Version checking
    std::cout << "Current version: " << compat::VersionInfo::CURRENT_VERSION << "\n";
    std::cout << "Minimum compatible: " << compat::VersionInfo::MIN_COMPATIBLE_VERSION << "\n";
    
    std::vector<std::string> test_versions = {"0.4.0", "0.3.0", "0.2.0", "0.1.0"};
    for (const auto& version : test_versions) {
        bool supported = compat::VersionInfo::is_supported(version);
        std::cout << "Version " << version << ": " << (supported ? "supported" : "not supported") << "\n";
    }
    
    // Model configuration
    compat::ModelConfig config_model;
    config_model.version = "0.4.0";
    config_model.model_type = "DemoModel";
    config_model.parameters["demo_param"] = 42.0f;
    
    std::string config_file = "/tmp/demo_model_config.json";
    if (compat::ModelSerializer::save_config(config_model, config_file)) {
        std::cout << "Model configuration saved to " << config_file << "\n";
        
        compat::ModelConfig loaded = compat::ModelSerializer::load_config(config_file);
        std::cout << "Loaded model type: " << loaded.model_type << "\n";
        std::cout << "Loaded model version: " << loaded.version << "\n";
    }
    
    std::cout << "\n=== Stage 5 Example Complete ===\n";
    std::cout << "Stage 5 successfully demonstrates:\n";
    std::cout << "✓ Experimental nodes (LIF, Add, RandomChoice)\n";
    std::cout << "✓ Hyperparameter optimization\n";
    std::cout << "✓ Plotting utilities with Python export\n";
    std::cout << "✓ Model compatibility and serialization\n";
    
    return 0;
}