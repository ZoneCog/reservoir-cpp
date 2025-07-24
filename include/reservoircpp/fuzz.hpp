#pragma once

#include <reservoircpp/types.hpp>
#include <random>
#include <vector>
#include <functional>
#include <string>

namespace reservoircpp {
namespace fuzz {

/**
 * @brief Fuzz testing framework for reservoir computing components
 */
class FuzzTester {
public:
    /**
     * @brief Test result structure
     */
    struct TestResult {
        std::string test_name;
        bool passed;
        std::string error_message;
        size_t iterations_completed;
    };

    /**
     * @brief Random input generator for matrices
     */
    class MatrixGenerator {
    public:
        MatrixGenerator(unsigned seed = 42) : rng(seed) {}
        
        /**
         * @brief Generate random matrix with specified constraints
         */
        Matrix random_matrix(int rows, int cols, float min_val = -10.0f, float max_val = 10.0f);
        
        /**
         * @brief Generate matrix with random dimensions
         */
        Matrix random_sized_matrix(int max_rows = 1000, int max_cols = 1000);
        
        /**
         * @brief Generate potentially problematic matrix (zeros, infinities, etc.)
         */
        Matrix problematic_matrix(int rows, int cols);
        
    private:
        std::mt19937 rng;
    };

    /**
     * @brief Parameter generator for fuzz testing
     */
    class ParameterGenerator {
    public:
        ParameterGenerator(unsigned seed = 42) : rng(seed) {}
        
        int random_int(int min_val, int max_val);
        float random_float(float min_val, float max_val);
        bool random_bool();
        std::string random_string(size_t length = 10);
        
    private:
        std::mt19937 rng;
    };

    FuzzTester(unsigned seed = 42) : matrix_gen(seed), param_gen(seed) {}
    
    /**
     * @brief Run comprehensive fuzz tests
     */
    std::vector<TestResult> run_all_fuzz_tests(size_t iterations_per_test = 1000);
    
    /**
     * @brief Fuzz test activation functions
     */
    std::vector<TestResult> fuzz_activations(size_t iterations = 1000);
    
    /**
     * @brief Fuzz test matrix generators
     */
    std::vector<TestResult> fuzz_matrix_generators(size_t iterations = 1000);
    
    /**
     * @brief Fuzz test reservoir operations
     */
    std::vector<TestResult> fuzz_reservoirs(size_t iterations = 1000);
    
    /**
     * @brief Fuzz test readout operations
     */
    std::vector<TestResult> fuzz_readouts(size_t iterations = 1000);
    
    /**
     * @brief Fuzz test dataset operations
     */
    std::vector<TestResult> fuzz_datasets(size_t iterations = 1000);
    
    /**
     * @brief Fuzz test observables/metrics
     */
    std::vector<TestResult> fuzz_observables(size_t iterations = 1000);

    /**
     * @brief Print test results summary
     */
    static void print_results(const std::vector<TestResult>& results);

    /**
     * @brief Safe execution wrapper that catches exceptions
     */
    TestResult safe_execute(const std::string& test_name, 
                           std::function<void()> test_func, 
                           size_t iterations);

private:
    MatrixGenerator matrix_gen;
    ParameterGenerator param_gen;
};

/**
 * @brief Input validation fuzz tester
 */
class InputValidationFuzzer {
public:
    /**
     * @brief Test boundary conditions for matrix operations
     */
    static std::vector<FuzzTester::TestResult> test_matrix_boundaries();
    
    /**
     * @brief Test invalid parameter ranges
     */
    static std::vector<FuzzTester::TestResult> test_parameter_ranges();
    
    /**
     * @brief Test memory allocation limits
     */
    static std::vector<FuzzTester::TestResult> test_memory_limits();
    
    /**
     * @brief Test numerical stability
     */
    static std::vector<FuzzTester::TestResult> test_numerical_stability();
};

/**
 * @brief Robustness tester for edge cases
 */
class RobustnessTester {
public:
    /**
     * @brief Test with extreme parameter values
     */
    static std::vector<FuzzTester::TestResult> test_extreme_parameters();
    
    /**
     * @brief Test with malformed inputs
     */
    static std::vector<FuzzTester::TestResult> test_malformed_inputs();
    
    /**
     * @brief Test concurrent access safety
     */
    static std::vector<FuzzTester::TestResult> test_thread_safety();
    
    /**
     * @brief Test resource cleanup
     */
    static std::vector<FuzzTester::TestResult> test_resource_cleanup();
};

} // namespace fuzz
} // namespace reservoircpp