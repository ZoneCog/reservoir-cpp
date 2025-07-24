# Migration Guide: Python ReservoirPy → C++ ReservoirCpp

This guide helps you migrate from the Python [ReservoirPy](https://github.com/reservoirpy/reservoirpy) library to the C++ ReservoirCpp implementation.

## Overview

ReservoirCpp provides a modern C++17 port of ReservoirPy with equivalent functionality and similar API design patterns. The migration maintains conceptual compatibility while leveraging C++ features for performance and type safety.

## Quick Comparison

| Aspect | Python ReservoirPy | C++ ReservoirCpp |
|--------|-------------------|------------------|
| **Performance** | NumPy-based | Eigen-based, compiled |
| **Type Safety** | Runtime checks | Compile-time + runtime |
| **Memory** | Garbage collected | RAII, explicit management |
| **Parallelism** | GIL limitations | Native threading |
| **Ecosystem** | Python ML stack | C++ ecosystem |

## Core Concepts Mapping

### 1. Basic Types and Arrays

**Python (ReservoirPy):**
```python
import numpy as np
import reservoirpy as rpy

# Arrays and matrices
data = np.random.rand(100, 5)
weights = np.random.uniform(-1, 1, (50, 50))
```

**C++ (ReservoirCpp):**
```cpp
#include <reservoircpp/reservoircpp.hpp>
using namespace reservoircpp;

// Matrices (Eigen-based)
Matrix data = utils::random_uniform(100, 5);
Matrix weights = matrix_generators::uniform(50, 50, -1.0, 1.0);
```

### 2. Activation Functions

**Python:**
```python
from reservoirpy.activationsfunc import tanh, sigmoid, softmax

# Apply activation functions
output = tanh(input_data)
prob = softmax(logits)
```

**C++:**
```cpp
#include <reservoircpp/activations.hpp>

// Get activation functions
auto tanh_fn = activations::get_function("tanh");
auto softmax_fn = activations::get_function("softmax");

// Apply activation functions
Matrix output = tanh_fn(input_data);
Matrix prob = softmax_fn(logits);
```

### 3. Nodes and Models

**Python:**
```python
from reservoirpy.nodes import Reservoir, Ridge

# Create nodes
reservoir = Reservoir(100, lr=0.3, sr=0.9)
readout = Ridge(ridge=1e-6)

# Connect nodes
model = reservoir >> readout
```

**C++:**
```cpp
#include <reservoircpp/reservoir.hpp>
#include <reservoircpp/readout.hpp>

// Create nodes
auto reservoir = std::make_unique<Reservoir>("reservoir", 100);
reservoir->set_parameter("lr", 0.3f);
reservoir->set_parameter("sr", 0.9f);

auto readout = std::make_unique<RidgeReadout>("readout", output_dim);
readout->set_parameter("ridge", 1e-6f);

// Manual connection (or use composition)
```

## Common Migration Patterns

### 1. Data Generation

**Python:**
```python
from reservoirpy.datasets import mackey_glass, to_forecasting

# Generate dataset
mg = mackey_glass(2000)
X, y = to_forecasting(mg, forecast=1)
```

**C++:**
```cpp
#include <reservoircpp/datasets.hpp>

// Generate dataset
auto mg = datasets::mackey_glass(2000);
auto [X, y] = datasets::to_forecasting(mg, 1);
```

### 2. Training Workflow

**Python:**
```python
# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.run(X_test)

# Evaluate
from reservoirpy.observables import nrmse
error = nrmse(y_test, y_pred)
```

**C++:**
```cpp
// Initialize and train
reservoir->initialize(&X_train);
auto states = reservoir->forward(X_train);
readout->fit(states, y_train);

// Make predictions
auto test_states = reservoir->forward(X_test);
auto y_pred = readout->forward(test_states);

// Evaluate
float error = observables::nrmse(y_test, y_pred);
```

### 3. Hyperparameter Optimization

**Python:**
```python
from reservoirpy.hyper import research

# Define search space
hyperopt_config = {
    "exp": "my_experiment",
    "hp_max_evals": 100,
    "hp_method": "random",
    "seed": 42
}

# Run optimization
best = research(dataset, model_func, hyperopt_config)
```

**C++:**
```cpp
#include <reservoircpp/hyper.hpp>

// Define search space
std::vector<hyper::ParameterSpace> space = {
    hyper::ParameterSpace::uniform("lr", 0.1f, 0.9f),
    hyper::ParameterSpace::uniform("sr", 0.1f, 1.5f)
};

// Run optimization
hyper::RandomSearch optimizer(space);
auto result = optimizer.optimize(objective_function, 100);
```

## Advanced Migration Topics

### 1. Custom Nodes

**Python:**
```python
from reservoirpy import Node

class CustomNode(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x):
        # Custom logic
        return processed_x
```

**C++ (Inherit from Node):**
```cpp
class CustomNode : public Node {
public:
    CustomNode(const std::string& name, int units) 
        : Node(name, units) {}
    
    Matrix forward(const Matrix& input) override {
        // Custom logic
        return processed_input;
    }
};
```

### 2. Memory Management

**Python (Automatic):**
```python
# Garbage collection handles cleanup
reservoir = Reservoir(100)
del reservoir  # Optional
```

**C++ (RAII):**
```cpp
// Automatic cleanup with smart pointers
auto reservoir = std::make_unique<Reservoir>("res", 100);
// Automatically cleaned up when out of scope

// Or manual management
Reservoir reservoir("res", 100);
// Automatically cleaned up when out of scope
```

### 3. Error Handling

**Python:**
```python
try:
    result = model.run(data)
except Exception as e:
    print(f"Error: {e}")
```

**C++:**
```cpp
try {
    auto result = model.forward(data);
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

## Performance Considerations

### 1. Compilation Optimization

Unlike Python, C++ requires compilation. Enable optimizations:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" ..
```

### 2. Memory Layout

C++ gives more control over memory:

```cpp
// Prefer stack allocation for small matrices
Matrix small_matrix(10, 10);

// Use move semantics for large data
Matrix large_data = std::move(generate_large_matrix());
```

### 3. Parallelization

C++ allows native threading without GIL:

```cpp
#include <execution>
#include <algorithm>

// Parallel algorithms (C++17)
std::transform(std::execution::par, 
               data.begin(), data.end(), 
               result.begin(), 
               activation_function);
```

## Migration Checklist

- [ ] **Dependencies**: Install Eigen3 and CMake
- [ ] **Build System**: Set up CMake configuration  
- [ ] **Data Types**: Convert NumPy arrays to Eigen matrices
- [ ] **Nodes**: Migrate node definitions and parameters
- [ ] **Training**: Adapt training loops and workflows
- [ ] **Evaluation**: Port metric calculations
- [ ] **Hyperparameters**: Convert optimization routines
- [ ] **Testing**: Validate numerical equivalence
- [ ] **Performance**: Profile and optimize hot paths

## Common Pitfalls

### 1. Matrix Indexing

**Python (0-based, row-major):**
```python
value = matrix[i, j]  # (row, col)
```

**C++ (0-based, same as Python):**
```cpp
float value = matrix(i, j);  // (row, col) - same semantics
```

### 2. Random Number Generation

**Python (NumPy):**
```python
np.random.seed(42)
data = np.random.rand(100, 50)
```

**C++ (Explicit seeding):**
```cpp
utils::set_seed(42);
Matrix data = matrix_generators::uniform(100, 50);
```

### 3. Broadcasting

Python NumPy has automatic broadcasting; C++ requires explicit operations:

**Python:**
```python
result = matrix + vector  # Automatic broadcasting
```

**C++ (Explicit):**
```cpp
Matrix result = matrix.rowwise() + vector.transpose();
```

## Example: Complete Migration

### Python Version
```python
import numpy as np
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import mackey_glass, to_forecasting
from reservoirpy.observables import nrmse

# Generate data
mg = mackey_glass(2000)
X, y = to_forecasting(mg, forecast=1)

# Create model
reservoir = Reservoir(100, lr=0.3, sr=0.9)
readout = Ridge(ridge=1e-6)
model = reservoir >> readout

# Split data
train_len = 1500
X_train, y_train = X[:train_len], y[:train_len]
X_test, y_test = X[train_len:], y[train_len:]

# Train and predict
model.fit(X_train, y_train)
y_pred = model.run(X_test)

# Evaluate
error = nrmse(y_test, y_pred)
print(f"NRMSE: {error}")
```

### C++ Version
```cpp
#include <reservoircpp/reservoircpp.hpp>
#include <iostream>

int main() {
    using namespace reservoircpp;
    
    // Generate data
    auto mg = datasets::mackey_glass(2000);
    auto [X, y] = datasets::to_forecasting(mg, 1);
    
    // Create model
    auto reservoir = std::make_unique<Reservoir>("reservoir", 100);
    reservoir->set_parameter("lr", 0.3f);
    reservoir->set_parameter("sr", 0.9f);
    
    auto readout = std::make_unique<RidgeReadout>("readout", 1);
    readout->set_parameter("ridge", 1e-6f);
    
    // Split data
    int train_len = 1500;
    Matrix X_train = X.topRows(train_len);
    Matrix y_train = y.topRows(train_len);
    Matrix X_test = X.bottomRows(X.rows() - train_len);
    Matrix y_test = y.bottomRows(y.rows() - train_len);
    
    // Train and predict
    reservoir->initialize(&X_train);
    auto states_train = reservoir->forward(X_train);
    readout->fit(states_train, y_train);
    
    auto states_test = reservoir->forward(X_test);
    auto y_pred = readout->forward(states_test);
    
    // Evaluate
    float error = observables::nrmse(y_test, y_pred);
    std::cout << "NRMSE: " << error << std::endl;
    
    return 0;
}
```

## Resources

- **API Reference**: [Documentation](https://zonecog.github.io/reservoircpp)
- **Examples**: Check `/examples` directory for comprehensive tutorials
- **Performance Guide**: See [PERFORMANCE.md](PERFORMANCE.md)
- **Community**: [GitHub Discussions](https://github.com/ZoneCog/reservoircpp/discussions)

## Getting Help

If you encounter issues during migration:

1. Check the [FAQ section](#faq) below
2. Compare with working examples in `/examples`
3. Open an issue with your Python code and attempted C++ conversion
4. Join the discussions for community support

## FAQ

**Q: Should I migrate everything at once?**
A: No, consider a gradual migration. Start with data processing, then core algorithms, finally advanced features.

**Q: How do I handle Python-specific libraries?**
A: Many Python ML libraries have C++ equivalents. For plotting, consider exporting data to Python or using matplotlib-cpp.

**Q: Is the numerical output identical?**
A: Very close but not bit-identical due to different underlying implementations. Validate with tolerance-based comparisons.

**Q: What about performance gains?**
A: Expect 5-50x speedup depending on your use case, especially for compute-intensive reservoir operations.

**Q: Can I use both libraries together?**
A: Yes! You can create Python bindings using pybind11 to call C++ code from Python if needed.