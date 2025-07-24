# ReservoirCpp User Guide

Welcome to ReservoirCpp, a modern C++17 implementation of reservoir computing algorithms. This guide provides comprehensive documentation for using the library effectively.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Best Practices](#best-practices)
6. [Performance Optimization](#performance-optimization)
7. [API Reference](#api-reference)

## Quick Start

### Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions. Quick setup:

```bash
# Install dependencies
sudo apt install cmake libeigen3-dev

# Clone and build
git clone https://github.com/ZoneCog/reservoircpp.git
cd reservoircpp
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
```

### Your First Reservoir

```cpp
#include <reservoircpp/reservoircpp.hpp>
#include <iostream>

int main() {
    using namespace reservoircpp;
    
    // Generate sample data
    auto data = datasets::mackey_glass(1000);
    auto [X, y] = datasets::to_forecasting(data, 1);
    
    // Create reservoir and readout
    auto reservoir = std::make_unique<Reservoir>("reservoir", 100);
    auto readout = std::make_unique<RidgeReadout>("readout", 1);
    
    // Split data
    int split = 800;
    Matrix X_train = X.topRows(split);
    Matrix y_train = y.topRows(split);
    Matrix X_test = X.bottomRows(X.rows() - split);
    Matrix y_test = y.bottomRows(y.rows() - split);
    
    // Train
    reservoir->initialize(&X_train);
    auto states = reservoir->forward(X_train);
    readout->fit(states, y_train);
    
    // Predict and evaluate
    auto test_states = reservoir->forward(X_test);
    auto y_pred = readout->forward(test_states);
    
    float mse = observables::mse(y_test, y_pred);
    std::cout << "MSE: " << mse << std::endl;
    
    return 0;
}
```

## Core Concepts

### 1. Nodes and Networks

ReservoirCpp is built around the concept of **Nodes** that can be connected to form networks:

- **Reservoir Nodes**: Process input and maintain internal state
- **Readout Nodes**: Learn from reservoir states to make predictions
- **Experimental Nodes**: Advanced components like spiking neurons

### 2. Data Flow

```
Input Data → Reservoir → Internal States → Readout → Output
```

### 3. Memory Management

ReservoirCpp uses modern C++ memory management:

```cpp
// Recommended: Use smart pointers
auto reservoir = std::make_unique<Reservoir>("name", 100);

// Or automatic cleanup with RAII
Reservoir reservoir("name", 100);
// Automatically cleaned up when out of scope
```

## Basic Usage

### Working with Datasets

```cpp
#include <reservoircpp/datasets.hpp>

// Generate chaotic time series
auto mg_data = datasets::mackey_glass(2000);          // Mackey-Glass
auto lorenz_data = datasets::lorenz(1000);            // Lorenz attractor
auto henon_data = datasets::henon_map(1500);          // Hénon map

// Convert to supervised learning format
auto [X, y] = datasets::to_forecasting(mg_data, 3);   // 3-step prediction

// One-hot encoding for discrete data
Matrix encoded = datasets::one_hot_encode(labels, num_classes);
```

### Creating and Configuring Reservoirs

```cpp
#include <reservoircpp/reservoir.hpp>

// Basic reservoir
auto reservoir = std::make_unique<Reservoir>("my_reservoir", 100);

// Configure parameters
reservoir->set_parameter("lr", 0.3f);           // Leaking rate
reservoir->set_parameter("sr", 0.9f);           // Spectral radius
reservoir->set_parameter("input_scaling", 1.0f); // Input scaling
reservoir->set_parameter("bias_scaling", 0.1f);  // Bias scaling

// Echo State Network with specific configuration
auto esn = std::make_unique<ESN>("esn", 200);
esn->set_parameter("lr", 0.1f);
esn->set_parameter("sr", 1.2f);

// NVAR reservoir for nonlinear tasks
auto nvar = std::make_unique<NVAR>("nvar", 50);
nvar->set_parameter("delay", 2);
nvar->set_parameter("order", 2);
```

### Working with Readouts

```cpp
#include <reservoircpp/readout.hpp>

// Ridge regression readout
auto ridge = std::make_unique<RidgeReadout>("ridge", output_dim);
ridge->set_parameter("ridge", 1e-6f);

// FORCE learning readout
auto force = std::make_unique<ForceReadout>("force", output_dim);
force->set_parameter("alpha", 1.0f);

// Online learning with LMS
auto lms = std::make_unique<LMSReadout>("lms", output_dim);
lms->set_parameter("learning_rate", 0.01f);
```

### Activation Functions

```cpp
#include <reservoircpp/activations.hpp>

// Get activation functions
auto tanh_fn = activations::get_function("tanh");
auto sigmoid_fn = activations::get_function("sigmoid");
auto relu_fn = activations::get_function("relu");

// Apply to data
Matrix activated = tanh_fn(input_data);

// Available functions: tanh, sigmoid, relu, softmax, identity, sign
```

### Matrix Generation

```cpp
#include <reservoircpp/matrix_generators.hpp>

// Generate random matrices
Matrix uniform_mat = matrix_generators::uniform(10, 10, -1.0, 1.0);
Matrix normal_mat = matrix_generators::normal(50, 50, 0.0, 1.0);
Matrix sparse_mat = matrix_generators::uniform(100, 100, -1.0, 1.0, 0.1); // 10% connectivity

// Generate reservoir weights
Matrix W = matrix_generators::generate_internal_weights(100, 0.1, 0.9);
Matrix Win = matrix_generators::generate_input_weights(100, 5, 1.0);
```

## Advanced Features

### 1. Experimental Nodes

```cpp
#include <reservoircpp/experimental.hpp>

// Leaky Integrate-and-Fire spiking neuron
experimental::LIF lif("spiking", 50);
lif.set_parameter("tau", 20.0f);
lif.set_parameter("threshold", 1.0f);
auto spikes = lif.forward(input);

// Utility nodes
experimental::Add adder("add");
Matrix sum = adder.forward({input1, input2});

// Random feature selection
experimental::RandomChoice selector("select", 10); // Select 10 features
selector.set_seed(42);
Matrix selected = selector.forward(input);
```

### 2. Hyperparameter Optimization

```cpp
#include <reservoircpp/hyper.hpp>

// Define search space
std::vector<hyper::ParameterSpace> search_space = {
    hyper::ParameterSpace::uniform("lr", 0.1f, 0.9f),
    hyper::ParameterSpace::uniform("sr", 0.1f, 1.5f),
    hyper::ParameterSpace::choice("units", {50, 100, 200, 500})
};

// Random search
hyper::RandomSearch random_opt(search_space);
auto random_result = random_opt.optimize(objective_function, 100);

// Grid search
hyper::GridSearch grid_opt(search_space);
auto grid_result = grid_opt.optimize(objective_function);

// Print best parameters
std::cout << "Best score: " << random_result.best_score << std::endl;
for (const auto& [param, value] : random_result.best_params) {
    std::cout << param << ": " << value << std::endl;
}
```

### 3. Model Persistence

```cpp
#include <reservoircpp/compat.hpp>

// Save model
compat::ModelSerializer::save_model(*reservoir, "my_model.bin");

// Load model  
auto loaded_reservoir = compat::ModelSerializer::load_model("my_model.bin");

// Export to Python format
compat::ModelSerializer::export_to_python(*reservoir, "python_export/");
```

### 4. Evaluation Metrics

```cpp
#include <reservoircpp/observables.hpp>

// Basic metrics
float mse = observables::mse(y_true, y_pred);
float rmse = observables::rmse(y_true, y_pred);
float nrmse = observables::nrmse(y_true, y_pred);
float r2 = observables::rsquare(y_true, y_pred);

// Reservoir-specific metrics
float sr = observables::spectral_radius(weight_matrix);
float esr = observables::effective_spectral_radius(reservoir_states);
float mc = observables::memory_capacity(reservoir, input_data);
```

## Best Practices

### 1. Parameter Tuning

```cpp
// Start with these defaults and tune
reservoir->set_parameter("lr", 0.3f);     // Leaking rate: 0.1-0.9
reservoir->set_parameter("sr", 0.9f);     // Spectral radius: 0.1-1.5
reservoir->set_parameter("input_scaling", 1.0f);  // Input scaling: 0.1-2.0

// For time series prediction, higher leaking rates often work better
// For memory tasks, lower leaking rates are preferred
```

### 2. Data Preprocessing

```cpp
// Normalize input data
Matrix normalized = (data.array() - data.mean()) / 
                   std::sqrt((data.array() - data.mean()).square().mean());

// Or use z-score normalization for each feature
for (int j = 0; j < data.cols(); ++j) {
    auto col = data.col(j);
    float mean = col.mean();
    float std = std::sqrt((col.array() - mean).square().mean());
    col = (col.array() - mean) / std;
}
```

### 3. Cross-Validation

```cpp
// K-fold cross-validation example
int k_folds = 5;
int fold_size = data.rows() / k_folds;
std::vector<float> scores;

for (int fold = 0; fold < k_folds; ++fold) {
    int test_start = fold * fold_size;
    int test_end = std::min(test_start + fold_size, (int)data.rows());
    
    // Split data
    Matrix X_train = /* training data excluding fold */;
    Matrix X_test = data.block(test_start, 0, test_end - test_start, data.cols());
    
    // Train and evaluate
    // ... training code ...
    float score = evaluate_model(X_test, y_test);
    scores.push_back(score);
}

float mean_score = std::accumulate(scores.begin(), scores.end(), 0.0f) / scores.size();
```

### 4. Memory Management

```cpp
// For large datasets, process in chunks
int chunk_size = 1000;
for (int i = 0; i < data.rows(); i += chunk_size) {
    int end = std::min(i + chunk_size, (int)data.rows());
    Matrix chunk = data.block(i, 0, end - i, data.cols());
    
    // Process chunk
    auto states = reservoir->forward(chunk);
    // ... continue processing ...
}

// Reset reservoir state between independent sequences
reservoir->reset();
```

## Performance Optimization

### 1. Compilation Flags

```bash
# Optimal performance build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native -DNDEBUG" \
      ..
```

### 2. Eigen Configuration

```cpp
// For better performance with large matrices
#define EIGEN_USE_MKL_ALL     // Use Intel MKL if available
#define EIGEN_USE_BLAS        // Use optimized BLAS
```

### 3. Profiling

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();
// ... code to profile ...
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
std::cout << "Time: " << duration.count() << " μs" << std::endl;
```

## API Reference

### Core Classes

- **`Node`**: Base class for all computational nodes
- **`Reservoir`**: Standard reservoir computing node  
- **`ESN`**: Echo State Network implementation
- **`NVAR`**: Nonlinear Autoregressive Vector reservoir
- **`RidgeReadout`**: Ridge regression readout
- **`ForceReadout`**: FORCE learning readout
- **`LMSReadout`**: Least Mean Squares readout

### Utility Namespaces

- **`datasets::`**: Data generation and preprocessing
- **`observables::`**: Evaluation metrics
- **`matrix_generators::`**: Matrix initialization
- **`activations::`**: Activation functions
- **`utils::`**: Utility functions
- **`experimental::`**: Advanced/experimental features
- **`hyper::`**: Hyperparameter optimization
- **`compat::`**: Model persistence and compatibility

### Key Types

```cpp
using Float = float;                          // Floating point type
using Matrix = Eigen::MatrixXf;              // Dense matrix
using Vector = Eigen::VectorXf;              // Dense vector
using SparseMatrix = Eigen::SparseMatrix<Float>; // Sparse matrix
```

## Examples and Tutorials

Check the `/examples` directory for comprehensive tutorials:

- `simple_example.cpp`: Basic reservoir usage
- `stage2_example.cpp`: Core reservoir computing
- `complete_tutorial.cpp`: Full workflow demonstration
- `stage6_tutorial.cpp`: Advanced features showcase

## Getting Help

- **Documentation**: [API Reference](https://zonecog.github.io/reservoircpp)
- **Examples**: See `/examples` directory
- **Issues**: [GitHub Issues](https://github.com/ZoneCog/reservoircpp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ZoneCog/reservoircpp/discussions)

## Next Steps

1. Try the examples in `/examples`
2. Read the [Migration Guide](MIGRATION.md) if coming from Python
3. Explore advanced features in the experimental module
4. Contribute to the project or report issues on GitHub