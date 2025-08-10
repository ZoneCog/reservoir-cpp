# Installation Guide for ReservoirCpp

ReservoirCpp is a modern C++17 implementation of reservoir computing algorithms. This guide covers various installation methods and configuration options.

## Prerequisites

- **C++17** compatible compiler (GCC 7+, Clang 6+, MSVC 2017+)
- **CMake 3.12+** for building
- **Eigen3 3.4+** for linear algebra

### Installing Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install build-essential cmake libeigen3-dev
```

#### macOS (with Homebrew)
```bash
brew install cmake eigen
```

#### Windows (with vcpkg)
```bash
vcpkg install eigen3
```

## Installation Methods

### 1. Header-Only Library (Recommended for Development)

For the simplest integration, use ReservoirCpp as a header-only library:

```bash
git clone https://github.com/ZoneCog/reservoircpp.git
cd reservoircpp
mkdir build && cd build
cmake -DBUILD_HEADER_ONLY=ON ..
make install
```

**Note:** The `make install` command must be run from the `build/` directory after running CMake, as it requires the generated Makefiles with the install target.

This installs only headers and makes the library available through CMake `find_package`.

### 2. Compiled Library (Recommended for Production)

For better compile times and smaller binaries:

```bash
git clone https://github.com/ZoneCog/reservoircpp.git
cd reservoircpp
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
```

**Note:** The `make install` command must be run from the `build/` directory after running CMake, as it requires the generated Makefiles with the install target.

### 3. Package Manager Installation

#### Using Conan
```bash
# Add to your conanfile.txt
[requires]
reservoircpp/0.1.0

# Or install directly
conan install reservoircpp/0.1.0@
```

#### Using vcpkg
```bash
vcpkg install reservoircpp
```

#### Using pkg-config (Linux/macOS)
After installing the compiled library:
```bash
pkg-config --cflags --libs reservoircpp
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_HEADER_ONLY` | OFF | Build as header-only library |
| `BUILD_SHARED_LIBS` | OFF | Build shared libraries instead of static |
| `BUILD_TESTS` | ON | Build test suite |
| `BUILD_EXAMPLES` | ON | Build example programs |
| `BUILD_DOCS` | OFF | Build documentation |

Example with custom options:
```bash
cmake -DBUILD_HEADER_ONLY=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF ..
```

## Using ReservoirCpp in Your Project

### CMake Integration

Add to your `CMakeLists.txt`:

```cmake
find_package(reservoircpp REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app reservoircpp::reservoircpp)
```

### Manual Integration

If not using CMake, include the headers and link against Eigen3:

```cpp
#include <reservoircpp/reservoircpp.hpp>
```

Compile with:
```bash
g++ -std=c++17 -I/path/to/reservoircpp/include -I/path/to/eigen3 main.cpp -o my_app
```

## Verification

Test your installation with this simple program:

```cpp
#include <reservoircpp/reservoircpp.hpp>
#include <iostream>

int main() {
    // Create a simple reservoir
    auto reservoir = std::make_unique<reservoircpp::Reservoir>("test", 100);
    
    // Generate some test data
    auto data = reservoircpp::datasets::mackey_glass(500);
    
    std::cout << "ReservoirCpp installed successfully!" << std::endl;
    std::cout << "Generated " << data.rows() << " samples of Mackey-Glass data" << std::endl;
    
    return 0;
}
```

## Troubleshooting

### Common Issues

1. **Eigen3 not found**: Make sure Eigen3 is installed and discoverable by CMake
   ```bash
   export CMAKE_PREFIX_PATH=/path/to/eigen3:$CMAKE_PREFIX_PATH
   ```

2. **C++17 not supported**: Update your compiler or specify the standard explicitly
   ```bash
   cmake -DCMAKE_CXX_STANDARD=17 ..
   ```

3. **Tests failing**: Some tests may fail on different architectures or with different compilers. This is usually harmless for library usage.

### Performance Optimization

For optimal performance, compile with:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" ..
```

### Debug Build

For development and debugging:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON ..
```

## Advanced Configuration

### Custom Eigen Location
```bash
cmake -DEigen3_DIR=/path/to/eigen3/cmake ..
```

### Cross-compilation
```bash
cmake -DCMAKE_TOOLCHAIN_FILE=path/to/toolchain.cmake ..
```

### Static Analysis Integration
```bash
cmake -DCMAKE_CXX_CLANG_TIDY="clang-tidy;-checks=*" ..
```

## Next Steps

- Read the [User Guide](USER_GUIDE.md) for detailed usage examples
- Check out the [Migration Guide](MIGRATION.md) for porting from Python ReservoirPy
- Explore the `/examples` directory for comprehensive tutorials
- Visit the [API Documentation](https://zonecog.github.io/reservoircpp) for detailed reference

## Support

- **Issues**: Report bugs at https://github.com/ZoneCog/reservoircpp/issues
- **Discussions**: Ask questions at https://github.com/ZoneCog/reservoircpp/discussions
- **Documentation**: https://zonecog.github.io/reservoircpp