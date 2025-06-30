# ReservoirCpp Documentation

## Overview

ReservoirCpp is a modern C++17 implementation of reservoir computing algorithms, providing a high-performance alternative to the Python ReservoirPy library.

## Architecture

The library follows a modular design with clear separation of concerns:

- **Core Types** (`include/reservoircpp/types.hpp`): Basic type definitions and interfaces
- **Nodes** (planned): Computational graph components
- **Reservoirs** (planned): Reservoir implementations (ESN, etc.)
- **Readouts** (planned): Output layer implementations
- **Matrix Generators** (planned): Weight matrix initialization
- **Activations** (planned): Activation functions

## Build System

The project uses CMake with modern C++17 features and the following dependencies:

- **Eigen3**: Linear algebra operations
- **Catch2**: Unit testing framework (automatically downloaded)

## Migration Status

This library is being developed following a multi-stage migration plan from ReservoirPy:

### ✅ Stage 0: Preparation and Planning (COMPLETED)
- [x] Project structure setup
- [x] CMake build system with Eigen3 integration
- [x] Basic type definitions equivalent to Python type.py
- [x] Testing framework with Catch2
- [x] CI-ready structure

### 🚧 Upcoming Stages
1. **Stage 1**: Core Framework and Data Structures
2. **Stage 2**: Reservoir Computing Components
3. **Stage 3**: Training and Evaluation
4. **Stage 4**: Data Handling
5. **Stage 5**: Advanced Features
6. **Stage 6**: Examples and Documentation
7. **Stage 7**: Testing and QA
8. **Stage 8**: Deployment and Packaging

## Getting Started

### Building

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Running Tests

```bash
cd build
ctest
```

## API Design Principles

1. **Modern C++17**: Utilizing latest language features for performance and safety
2. **Header-only when possible**: Minimize compilation dependencies
3. **Eigen integration**: Leverage proven linear algebra library
4. **Type safety**: Strong typing to prevent runtime errors
5. **Memory efficiency**: Smart pointers and RAII principles