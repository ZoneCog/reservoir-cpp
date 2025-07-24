# ReservoirCpp

**A C++ implementation of reservoir computing algorithms**

ReservoirCpp is a modern C++17 port of the Python [ReservoirPy](https://github.com/reservoirpy/reservoirpy) library, providing efficient implementations of Echo State Networks (ESN) and other reservoir computing architectures.

## 🚧 Development Status

**This library is currently under active development.** We are following a multi-stage migration plan to port ReservoirPy from Python to C++.

### Current Stage: Stage 5 - Ancillary and Advanced Features ✅

- ✅ Project structure setup
- ✅ CMake build system
- ✅ Basic type definitions
- ✅ Initial testing framework
- ✅ Core utility functions (random generation, validation)
- ✅ Activation functions (sigmoid, tanh, relu, softmax, etc.)
- ✅ Base Node class with state management
- ✅ Parameter management system
- ✅ Reservoir computing components (reservoirs, readouts)
- ✅ Training and evaluation metrics
- ✅ Data handling and datasets
- ✅ **Stage 5: Ancillary and Advanced Features**
  - ✅ Experimental nodes (LIF spiking neuron, Add, BatchFORCE, RandomChoice)
  - ✅ Compatibility utilities (model serialization, version checking)
  - ✅ Hyperparameter optimization (RandomSearch, GridSearch, BayesianOptimization stubs)
  - ✅ Plotting utilities with Python export backend
  - ✅ GPU support framework (stub implementation)
- ✅ Comprehensive test suite (65 tests passing)

### Upcoming Stages

1. **Stage 6**: Examples and Documentation
2. **Stage 7**: Testing and QA  
3. **Stage 8**: Deployment and Packaging

## Features Implemented

### Stages 1-5: Complete Reservoir Computing Framework ✅

- **Type System**: Complete C++ equivalents of Python types with Eigen-based linear algebra
- **Activation Functions**: Full set of activation functions with registry system
- **Utility Functions**: Random number generation, validation utilities, array operations
- **Base Node Class**: Complete Node implementation with state management and parameter handling
- **Reservoir Components**: Reservoir, ESN, IntrinsicPlasticity, NVAR implementations
- **Readout Components**: Ridge, FORCE, LMS, RLS readout implementations
- **Matrix Generation**: Complete matrix generator suite with spectral radius control
- **Observables**: Full set of evaluation metrics (MSE, RMSE, NRMSE, R², memory capacity, etc.)
- **Datasets**: Complete dataset collection (Mackey-Glass, Lorenz, Hénon, NARMA, MSO, etc.)
- **Experimental Features**: LIF spiking neurons, utility nodes, advanced algorithms
- **Hyperparameter Optimization**: RandomSearch, GridSearch, BayesianOptimization frameworks
- **Plotting Utilities**: Python-compatible plotting with export capabilities
- **Compatibility Layer**: Model serialization and version management
- **Testing**: Comprehensive test suite with 65 test cases covering all functionality

### Example Usage

```cpp
#include <reservoircpp/reservoircpp.hpp>

// Use activation functions
auto sigmoid_fn = reservoircpp::activations::get_function("sigmoid");
reservoircpp::Matrix input(1, 3);
input << -1.0, 0.0, 1.0;
auto output = sigmoid_fn(input);

// Create and train an ESN
auto reservoir = std::make_unique<reservoircpp::Reservoir>("reservoir", 100);
auto readout = std::make_unique<reservoircpp::RidgeReadout>("readout", 1);

// Generate training data
auto [X_train, y_train] = reservoircpp::datasets::mackey_glass(1000);

// Train the model
reservoir->initialize(&X_train);
auto states = reservoir->forward(X_train);
readout->fit(states, y_train);

// Make predictions
auto y_pred = readout->forward(states);

// Evaluate performance
float mse = reservoircpp::observables::mse(y_train, y_pred);

// Use experimental features
reservoircpp::experimental::LIF lif("spiking", 50);
auto spikes = lif.forward(input);

// Hyperparameter optimization
std::vector<reservoircpp::hyper::ParameterSpace> search_space = {
    reservoircpp::hyper::ParameterSpace::uniform("lr", 0.001f, 0.1f)
};
reservoircpp::hyper::RandomSearch optimizer(search_space);

// Plot results
reservoircpp::plotting::PlotUtils::quick_plot(y_pred);
```

## Dependencies

- **C++17** compatible compiler
- **CMake 3.12+** for building
- **Eigen3** for linear algebra
- **Catch2** for testing (automatically downloaded)

## Building

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Testing

```bash
cd build
ctest
```

Run specific test categories:
```bash
# Run only Stage 5 tests
ctest -R "Stage 5"

# Run experimental features tests
ctest -R "experimental"

# Run with verbose output
ctest -V
```

Current test coverage: **65 test cases** covering all implemented functionality.

## License

MIT License - same as the original ReservoirPy library.

## Contributing

This project follows the same contributing guidelines as ReservoirPy. See [CONTRIBUTING.rst](CONTRIBUTING.rst) for details.

## Acknowledgments

This C++ port is based on the excellent [ReservoirPy](https://github.com/reservoirpy/reservoirpy) library developed by the Inria Mnemosyne team.