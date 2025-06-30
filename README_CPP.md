# ReservoirCpp

**A C++ implementation of reservoir computing algorithms**

ReservoirCpp is a modern C++17 port of the Python [ReservoirPy](https://github.com/reservoirpy/reservoirpy) library, providing efficient implementations of Echo State Networks (ESN) and other reservoir computing architectures.

## 🚧 Development Status

**This library is currently under active development.** We are following a multi-stage migration plan to port ReservoirPy from Python to C++.

### Current Stage: Stage 0 - Preparation and Planning ✅

- ✅ Project structure setup
- ✅ CMake build system
- ✅ Basic type definitions
- ✅ Initial testing framework
- ⏳ Additional core infrastructure

### Upcoming Stages

1. **Stage 1**: Core Framework and Data Structures
2. **Stage 2**: Reservoir Computing Components  
3. **Stage 3**: Training and Evaluation
4. **Stage 4**: Data Handling
5. **Stage 5**: Advanced Features
6. **Stage 6**: Examples and Documentation
7. **Stage 7**: Testing and QA
8. **Stage 8**: Deployment and Packaging

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