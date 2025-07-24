#include <reservoircpp/fuzz.hpp>
#include <reservoircpp/reservoircpp.hpp>
#include <iostream>
#include <sstream>
#include <limits>
#include <cmath>

namespace reservoircpp {
namespace fuzz {

Matrix FuzzTester::MatrixGenerator::random_matrix(int rows, int cols, float min_val, float max_val) {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    Matrix m(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            m(i, j) = dist(rng);
        }
    }
    return m;
}

Matrix FuzzTester::MatrixGenerator::random_sized_matrix(int max_rows, int max_cols) {
    std::uniform_int_distribution<int> row_dist(1, max_rows);
    std::uniform_int_distribution<int> col_dist(1, max_cols);
    int rows = row_dist(rng);
    int cols = col_dist(rng);
    return random_matrix(rows, cols);
}

Matrix FuzzTester::MatrixGenerator::problematic_matrix(int rows, int cols) {
    std::uniform_int_distribution<int> type_dist(0, 4);
    Matrix m(rows, cols);
    
    int type = type_dist(rng);
    switch (type) {
        case 0: // All zeros
            m.setZero();
            break;
        case 1: // All ones
            m.setOnes();
            break;
        case 2: // Very large values
            m.setConstant(1e6f);
            break;
        case 3: // Very small values
            m.setConstant(1e-6f);
            break;
        case 4: // Mixed with some NaN/inf
            m = random_matrix(rows, cols);
            if (rows > 0 && cols > 0) {
                m(0, 0) = std::numeric_limits<float>::infinity();
                if (rows > 1 && cols > 1) {
                    m(1, 1) = std::numeric_limits<float>::quiet_NaN();
                }
            }
            break;
    }
    return m;
}

int FuzzTester::ParameterGenerator::random_int(int min_val, int max_val) {
    std::uniform_int_distribution<int> dist(min_val, max_val);
    return dist(rng);
}

float FuzzTester::ParameterGenerator::random_float(float min_val, float max_val) {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    return dist(rng);
}

bool FuzzTester::ParameterGenerator::random_bool() {
    std::uniform_int_distribution<int> dist(0, 1);
    return dist(rng) == 1;
}

std::string FuzzTester::ParameterGenerator::random_string(size_t length) {
    const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::uniform_int_distribution<int> dist(0, chars.size() - 1);
    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        result += chars[dist(rng)];
    }
    return result;
}

FuzzTester::TestResult FuzzTester::safe_execute(const std::string& test_name, 
                                               std::function<void()> test_func, 
                                               size_t iterations) {
    TestResult result;
    result.test_name = test_name;
    result.passed = true;
    result.iterations_completed = 0;
    
    try {
        for (size_t i = 0; i < iterations; ++i) {
            test_func();
            result.iterations_completed++;
        }
    } catch (const std::exception& e) {
        result.passed = false;
        result.error_message = e.what();
    } catch (...) {
        result.passed = false;
        result.error_message = "Unknown exception";
    }
    
    return result;
}

std::vector<FuzzTester::TestResult> FuzzTester::run_all_fuzz_tests(size_t iterations_per_test) {
    std::vector<TestResult> all_results;
    
    auto activation_results = fuzz_activations(iterations_per_test);
    auto matrix_gen_results = fuzz_matrix_generators(iterations_per_test);
    auto reservoir_results = fuzz_reservoirs(iterations_per_test);
    auto readout_results = fuzz_readouts(iterations_per_test);
    auto dataset_results = fuzz_datasets(iterations_per_test);
    auto observable_results = fuzz_observables(iterations_per_test);
    
    all_results.insert(all_results.end(), activation_results.begin(), activation_results.end());
    all_results.insert(all_results.end(), matrix_gen_results.begin(), matrix_gen_results.end());
    all_results.insert(all_results.end(), reservoir_results.begin(), reservoir_results.end());
    all_results.insert(all_results.end(), readout_results.begin(), readout_results.end());
    all_results.insert(all_results.end(), dataset_results.begin(), dataset_results.end());
    all_results.insert(all_results.end(), observable_results.begin(), observable_results.end());
    
    return all_results;
}

std::vector<FuzzTester::TestResult> FuzzTester::fuzz_activations(size_t iterations) {
    std::vector<TestResult> results;
    
    // Test sigmoid with random inputs
    auto sigmoid_test = safe_execute("Sigmoid Fuzz Test", [this]() {
        Matrix input = matrix_gen.random_matrix(
            param_gen.random_int(1, 100), 
            param_gen.random_int(1, 100), 
            -100.0f, 100.0f);
        auto sigmoid_fn = reservoircpp::activations::get_function("sigmoid");
        auto output = sigmoid_fn(input);
        
        // Check output is in valid range [0, 1]
        for (int i = 0; i < output.rows(); ++i) {
            for (int j = 0; j < output.cols(); ++j) {
                float val = output(i, j);
                if (std::isnan(val) || val < 0.0f || val > 1.0f) {
                    throw std::runtime_error("Sigmoid output out of range");
                }
            }
        }
    }, iterations);
    results.push_back(sigmoid_test);
    
    // Test tanh with random inputs
    auto tanh_test = safe_execute("Tanh Fuzz Test", [this]() {
        Matrix input = matrix_gen.random_matrix(
            param_gen.random_int(1, 100), 
            param_gen.random_int(1, 100), 
            -100.0f, 100.0f);
        auto tanh_fn = reservoircpp::activations::get_function("tanh");
        auto output = tanh_fn(input);
        
        // Check output is in valid range [-1, 1]
        for (int i = 0; i < output.rows(); ++i) {
            for (int j = 0; j < output.cols(); ++j) {
                float val = output(i, j);
                if (std::isnan(val) || val < -1.0f || val > 1.0f) {
                    throw std::runtime_error("Tanh output out of range");
                }
            }
        }
    }, iterations);
    results.push_back(tanh_test);
    
    // Test ReLU with random inputs
    auto relu_test = safe_execute("ReLU Fuzz Test", [this]() {
        Matrix input = matrix_gen.random_matrix(
            param_gen.random_int(1, 100), 
            param_gen.random_int(1, 100), 
            -100.0f, 100.0f);
        auto relu_fn = reservoircpp::activations::get_function("relu");
        auto output = relu_fn(input);
        
        // Check output is non-negative
        for (int i = 0; i < output.rows(); ++i) {
            for (int j = 0; j < output.cols(); ++j) {
                float val = output(i, j);
                if (std::isnan(val) || val < 0.0f) {
                    throw std::runtime_error("ReLU output negative or NaN");
                }
            }
        }
    }, iterations);
    results.push_back(relu_test);
    
    return results;
}

std::vector<FuzzTester::TestResult> FuzzTester::fuzz_matrix_generators(size_t iterations) {
    std::vector<TestResult> results;
    
    // Test random matrix generation
    auto random_gen_test = safe_execute("Random Matrix Generation Fuzz Test", [this]() {
        int rows = param_gen.random_int(1, 500);
        float connectivity = param_gen.random_float(0.01f, 1.0f);
        float spectral_radius = param_gen.random_float(0.1f, 2.0f);
        
        auto matrix = reservoircpp::matrix_generators::generate_internal_weights(
            rows, connectivity, spectral_radius);
        
        if (matrix.rows() != rows || matrix.cols() != rows) {
            throw std::runtime_error("Matrix dimensions incorrect");
        }
    }, iterations);
    results.push_back(random_gen_test);
    
    return results;
}

std::vector<FuzzTester::TestResult> FuzzTester::fuzz_reservoirs(size_t iterations) {
    std::vector<TestResult> results;
    
    // Test reservoir creation with random parameters
    auto reservoir_creation_test = safe_execute("Reservoir Creation Fuzz Test", [this]() {
        int units = param_gen.random_int(1, 1000);
        std::string name = param_gen.random_string(10);
        
        Reservoir reservoir(name, units);
        
        auto output_shape = reservoir.output_dim();
        if (output_shape.empty() || output_shape[0] != units) {
            throw std::runtime_error("Reservoir units count incorrect");
        }
    }, iterations);
    results.push_back(reservoir_creation_test);
    
    // Test reservoir forward pass with random inputs
    auto reservoir_forward_test = safe_execute("Reservoir Forward Pass Fuzz Test", [this]() {
        int units = param_gen.random_int(10, 200);
        int input_dim = param_gen.random_int(1, 50);
        int time_steps = param_gen.random_int(1, 100);
        
        Reservoir reservoir("test", units);
        Matrix input = matrix_gen.random_matrix(time_steps, input_dim, -10.0f, 10.0f);
        
        reservoir.initialize(&input);
        auto states = reservoir.forward(input);
        
        if (states.rows() != time_steps || states.cols() != units) {
            throw std::runtime_error("Reservoir output dimensions incorrect");
        }
    }, iterations);
    results.push_back(reservoir_forward_test);
    
    return results;
}

std::vector<FuzzTester::TestResult> FuzzTester::fuzz_readouts(size_t iterations) {
    std::vector<TestResult> results;
    
    // Test Ridge readout with random data
    auto ridge_test = safe_execute("Ridge Readout Fuzz Test", [this]() {
        int n_samples = param_gen.random_int(10, 500);
        int n_features = param_gen.random_int(1, 100);
        int n_outputs = param_gen.random_int(1, 20);
        
        Matrix states = matrix_gen.random_matrix(n_samples, n_features, -10.0f, 10.0f);
        Matrix targets = matrix_gen.random_matrix(n_samples, n_outputs, -10.0f, 10.0f);
        
        RidgeReadout readout("test", n_outputs);
        readout.fit(states, targets);
        
        auto predictions = readout.forward(states);
        if (predictions.rows() != n_samples || predictions.cols() != n_outputs) {
            throw std::runtime_error("Ridge readout output dimensions incorrect");
        }
    }, iterations);
    results.push_back(ridge_test);
    
    return results;
}

std::vector<FuzzTester::TestResult> FuzzTester::fuzz_datasets(size_t iterations) {
    std::vector<TestResult> results;
    
    // Test Mackey-Glass with random parameters
    auto mackey_glass_test = safe_execute("Mackey-Glass Fuzz Test", [this]() {
        int n_timesteps = param_gen.random_int(100, 5000);
        auto data = reservoircpp::datasets::mackey_glass(n_timesteps);
        
        if (data.rows() != n_timesteps) {
            throw std::runtime_error("Mackey-Glass output dimensions incorrect");
        }
    }, iterations);
    results.push_back(mackey_glass_test);
    
    return results;
}

std::vector<FuzzTester::TestResult> FuzzTester::fuzz_observables(size_t iterations) {
    std::vector<TestResult> results;
    
    // Test MSE with random data
    auto mse_test = safe_execute("MSE Fuzz Test", [this]() {
        int n_samples = param_gen.random_int(1, 1000);
        int n_features = param_gen.random_int(1, 50);
        
        Matrix y_true = matrix_gen.random_matrix(n_samples, n_features, -10.0f, 10.0f);
        Matrix y_pred = matrix_gen.random_matrix(n_samples, n_features, -10.0f, 10.0f);
        
        float mse = reservoircpp::observables::mse(y_true, y_pred);
        
        if (std::isnan(mse) || mse < 0.0f) {
            throw std::runtime_error("MSE invalid result");
        }
    }, iterations);
    results.push_back(mse_test);
    
    return results;
}

void FuzzTester::print_results(const std::vector<TestResult>& results) {
    std::cout << "\n=== FUZZ TEST RESULTS ===" << std::endl;
    
    int passed = 0;
    int failed = 0;
    
    for (const auto& result : results) {
        std::cout << "Test: " << result.test_name << std::endl;
        std::cout << "  Status: " << (result.passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "  Iterations: " << result.iterations_completed << std::endl;
        if (!result.passed) {
            std::cout << "  Error: " << result.error_message << std::endl;
        }
        std::cout << std::endl;
        
        if (result.passed) {
            passed++;
        } else {
            failed++;
        }
    }
    
    std::cout << "Summary: " << passed << " passed, " << failed << " failed" << std::endl;
}

// Input validation fuzzer implementations
std::vector<FuzzTester::TestResult> InputValidationFuzzer::test_matrix_boundaries() {
    std::vector<FuzzTester::TestResult> results;
    
    FuzzTester fuzzer;
    
    // Test zero-sized matrices
    auto zero_size_test = fuzzer.safe_execute("Zero Size Matrix Test", []() {
        try {
            Matrix m(0, 0);
            // Should handle gracefully
        } catch (const std::exception&) {
            // Expected for some operations
        }
    }, 10);
    results.push_back(zero_size_test);
    
    return results;
}

std::vector<FuzzTester::TestResult> InputValidationFuzzer::test_parameter_ranges() {
    std::vector<FuzzTester::TestResult> results;
    
    FuzzTester fuzzer;
    
    // Test negative dimensions
    auto negative_dim_test = fuzzer.safe_execute("Negative Dimension Test", []() {
        try {
            Reservoir reservoir("test", -1);
            throw std::runtime_error("Should have thrown exception for negative dimension");
        } catch (const std::invalid_argument&) {
            // Expected
        }
    }, 10);
    results.push_back(negative_dim_test);
    
    return results;
}

std::vector<FuzzTester::TestResult> InputValidationFuzzer::test_memory_limits() {
    std::vector<FuzzTester::TestResult> results;
    
    FuzzTester fuzzer;
    
    // Test very large matrix allocation (should fail gracefully)
    auto large_matrix_test = fuzzer.safe_execute("Large Matrix Test", []() {
        try {
            // Try to allocate a very large matrix
            Matrix large_matrix(100000, 100000);
            // If this succeeds, that's also fine
        } catch (const std::bad_alloc&) {
            // Expected for large allocations
        }
    }, 1);
    results.push_back(large_matrix_test);
    
    return results;
}

std::vector<FuzzTester::TestResult> InputValidationFuzzer::test_numerical_stability() {
    std::vector<FuzzTester::TestResult> results;
    
    FuzzTester fuzzer;
    
    // Test with very small/large numbers
    auto numerical_test = fuzzer.safe_execute("Numerical Stability Test", []() {
        Matrix small_vals(10, 10);
        small_vals.setConstant(1e-30f);
        Matrix large_vals(10, 10);
        large_vals.setConstant(1e30f);
        
        auto sigmoid_fn = reservoircpp::activations::get_function("sigmoid");
        auto small_result = sigmoid_fn(small_vals);
        auto large_result = sigmoid_fn(large_vals);
        
        // Check for inf/nan
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 10; ++j) {
                if (std::isnan(small_result(i, j)) || std::isinf(small_result(i, j)) ||
                    std::isnan(large_result(i, j)) || std::isinf(large_result(i, j))) {
                    throw std::runtime_error("Numerical instability detected");
                }
            }
        }
    }, 100);
    results.push_back(numerical_test);
    
    return results;
}

} // namespace fuzz
} // namespace reservoircpp