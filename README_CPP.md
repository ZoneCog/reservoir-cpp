# ReservoirCpp

**A C++ implementation of reservoir computing algorithms**

ReservoirCpp is a modern C++17 port of the Python [ReservoirPy](https://github.com/reservoirpy/reservoirpy) library, providing efficient implementations of Echo State Networks (ESN) and other reservoir computing architectures.

## 🚧 Development Status

**This library is currently under active development.** We are following a multi-stage migration plan to port ReservoirPy from Python to C++.

### Current Stage: Stage 1 - Core Framework and Data Structures ✅

- ✅ Project structure setup
- ✅ CMake build system
- ✅ Basic type definitions
- ✅ Initial testing framework
- ✅ Core utility functions (random generation, validation)
- ✅ Activation functions (sigmoid, tanh, relu, softmax, etc.)
- ✅ Base Node class with state management
- ✅ Parameter management system
- ✅ Comprehensive test suite (17 tests passing)

### Upcoming Stages

1. **Stage 2**: Reservoir Computing Components  
2. **Stage 3**: Training and Evaluation
3. **Stage 4**: Data Handling
4. **Stage 5**: Advanced Features
5. **Stage 6**: Examples and Documentation
6. **Stage 7**: Testing and QA
7. **Stage 8**: Deployment and Packaging

## Features Implemented

### Stage 1: Core Framework and Data Structures ✅

- **Type System**: Complete C++ equivalents of Python types with Eigen-based linear algebra
- **Activation Functions**: Full set of activation functions (sigmoid, tanh, relu, softmax, softplus, identity) with registry system
- **Utility Functions**: Random number generation, validation utilities, array operations
- **Base Node Class**: Complete Node implementation with state management, parameter handling, and initialization
- **Testing**: Comprehensive test suite with 17 test cases covering all implemented functionality

### Example Usage

```cpp
#include <reservoircpp/reservoircpp.hpp>

// Use activation functions
auto sigmoid_fn = reservoircpp::activations::get_function("sigmoid");
reservoircpp::Matrix input(1, 3);
input << -1.0, 0.0, 1.0;
auto output = sigmoid_fn(input);

// Create and use nodes
reservoircpp::Node node("my_node");
node.set_output_dim({10});
node.initialize();
auto result = node(input);
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

## License

MIT License - same as the original ReservoirPy library.

## Contributing

This project follows the same contributing guidelines as ReservoirPy. See [CONTRIBUTING.rst](CONTRIBUTING.rst) for details.

## Acknowledgments

This C++ port is based on the excellent [ReservoirPy](https://github.com/reservoirpy/reservoirpy) library developed by the Inria Mnemosyne team.