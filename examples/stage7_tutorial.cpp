#include <reservoircpp/reservoircpp.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace reservoircpp;

/**
 * @brief Stage 7 - Testing and Quality Assurance Tutorial
 * 
 * This tutorial demonstrates the comprehensive testing and quality assurance
 * features introduced in Stage 7 of the py2cpp migration.
 */
int main() {
    std::cout << "ReservoirCpp Stage 7 - Testing and Quality Assurance Tutorial" << std::endl;
    std::cout << reservoircpp::version_info() << std::endl;
    std::cout << std::endl;

    try {
        // ==================================================
        // PERFORMANCE BENCHMARKING
        // ==================================================
        std::cout << "==================================================" << std::endl;
        std::cout << "  PERFORMANCE BENCHMARKING" << std::endl;
        std::cout << "==================================================" << std::endl;
        
        std::cout << "Running comprehensive performance benchmarks..." << std::endl;
        
        // Run matrix operation benchmarks
        std::cout << "\n--- Matrix Operations Benchmarks ---" << std::endl;
        auto matrix_results = benchmark::ReservoirBenchmarks::benchmark_matrix_operations();
        for (const auto& result : matrix_results) {
            benchmark::BenchmarkTimer::print_result(result);
        }
        
        // Run activation function benchmarks
        std::cout << "\n--- Activation Functions Benchmarks ---" << std::endl;
        auto activation_results = benchmark::ReservoirBenchmarks::benchmark_activations();
        for (const auto& result : activation_results) {
            benchmark::BenchmarkTimer::print_result(result);
        }
        
        // Run reservoir benchmarks
        std::cout << "\n--- Reservoir Operations Benchmarks ---" << std::endl;
        auto reservoir_results = benchmark::ReservoirBenchmarks::benchmark_reservoirs();
        for (const auto& result : reservoir_results) {
            benchmark::BenchmarkTimer::print_result(result);
        }
        
        // ==================================================
        // MEMORY PROFILING
        // ==================================================
        std::cout << "==================================================" << std::endl;
        std::cout << "  MEMORY PROFILING" << std::endl;
        std::cout << "==================================================" << std::endl;
        
        std::cout << "Profiling memory usage of typical operations..." << std::endl;
        
        auto [mem_before, mem_after] = benchmark::MemoryProfiler::profile_memory([]() {
            // Create a typical reservoir computing setup
            Reservoir reservoir("memory_test", 500);
            RidgeReadout readout("memory_test", 10);
            
            Matrix input = Matrix::Random(1000, 20);
            Matrix targets = Matrix::Random(1000, 10);
            
            reservoir.initialize(&input);
            auto states = reservoir.forward(input);
            readout.fit(states, targets);
            auto predictions = readout.forward(states);
        });
        
        if (mem_before > 0 && mem_after > 0) {
            std::cout << "Memory usage - Before: " << mem_before << " bytes" << std::endl;
            std::cout << "Memory usage - After: " << mem_after << " bytes" << std::endl;
            std::cout << "Memory increase: " << (mem_after - mem_before) << " bytes" << std::endl;
        } else {
            std::cout << "Memory profiling not available on this platform" << std::endl;
        }
        
        // ==================================================
        // FUZZ TESTING
        // ==================================================
        std::cout << "\n==================================================" << std::endl;
        std::cout << "  FUZZ TESTING" << std::endl;
        std::cout << "==================================================" << std::endl;
        
        std::cout << "Running fuzz tests to validate robustness..." << std::endl;
        
        fuzz::FuzzTester fuzzer(42); // Fixed seed for reproducibility
        
        // Test activation functions
        std::cout << "\n--- Activation Functions Fuzz Testing ---" << std::endl;
        auto activation_fuzz = fuzzer.fuzz_activations(100);
        fuzz::FuzzTester::print_results(activation_fuzz);
        
        // Test matrix generators
        std::cout << "\n--- Matrix Generators Fuzz Testing ---" << std::endl;
        auto matrix_fuzz = fuzzer.fuzz_matrix_generators(50);
        fuzz::FuzzTester::print_results(matrix_fuzz);
        
        // Test reservoirs
        std::cout << "\n--- Reservoir Operations Fuzz Testing ---" << std::endl;
        auto reservoir_fuzz = fuzzer.fuzz_reservoirs(50);
        fuzz::FuzzTester::print_results(reservoir_fuzz);
        
        // ==================================================
        // INPUT VALIDATION TESTING
        // ==================================================
        std::cout << "\n==================================================" << std::endl;
        std::cout << "  INPUT VALIDATION TESTING" << std::endl;
        std::cout << "==================================================" << std::endl;
        
        std::cout << "Testing input validation and boundary conditions..." << std::endl;
        
        // Test boundary conditions
        auto boundary_tests = fuzz::InputValidationFuzzer::test_matrix_boundaries();
        auto param_tests = fuzz::InputValidationFuzzer::test_parameter_ranges();
        auto memory_tests = fuzz::InputValidationFuzzer::test_memory_limits();
        auto numerical_tests = fuzz::InputValidationFuzzer::test_numerical_stability();
        
        std::vector<fuzz::FuzzTester::TestResult> all_validation_tests;
        all_validation_tests.insert(all_validation_tests.end(), boundary_tests.begin(), boundary_tests.end());
        all_validation_tests.insert(all_validation_tests.end(), param_tests.begin(), param_tests.end());
        all_validation_tests.insert(all_validation_tests.end(), memory_tests.begin(), memory_tests.end());
        all_validation_tests.insert(all_validation_tests.end(), numerical_tests.begin(), numerical_tests.end());
        
        fuzz::FuzzTester::print_results(all_validation_tests);
        
        // ==================================================
        // QUALITY ASSURANCE CHECKS
        // ==================================================
        std::cout << "\n==================================================" << std::endl;
        std::cout << "  QUALITY ASSURANCE CHECKS" << std::endl;
        std::cout << "==================================================" << std::endl;
        
        std::cout << "Running quality assurance checks..." << std::endl;
        
        // Test numerical stability
        std::cout << "\n--- Numerical Stability Check ---" << std::endl;
        Matrix extreme_values(5, 5);
        extreme_values << 1e6, -1e6, 1e-6, -1e-6, 0,
                         1e10, -1e10, 1e-10, -1e-10, 1,
                         1e20, -1e20, 1e-20, -1e-20, 2,
                         1e30, -1e30, 1e-30, -1e-30, 3,
                         1e35, -1e35, 1e-35, -1e-35, 4;
        
        auto sigmoid_fn = activations::get_function("sigmoid");
        auto tanh_fn = activations::get_function("tanh");
        auto relu_fn = activations::get_function("relu");
        
        auto sigmoid_result = sigmoid_fn(extreme_values);
        auto tanh_result = tanh_fn(extreme_values);
        auto relu_result = relu_fn(extreme_values);
        
        bool numerical_stable = true;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                if (!std::isfinite(sigmoid_result(i, j)) || 
                    !std::isfinite(tanh_result(i, j)) || 
                    !std::isfinite(relu_result(i, j))) {
                    numerical_stable = false;
                }
            }
        }
        
        std::cout << "Numerical stability: " << (numerical_stable ? "✓ PASS" : "✗ FAIL") << std::endl;
        
        // Test reproducibility
        std::cout << "\n--- Reproducibility Check ---" << std::endl;
        utils::set_seed(123);
        auto mg1 = datasets::mackey_glass(500);
        auto data1 = datasets::to_forecasting(mg1);
        auto X1 = std::get<0>(data1);
        auto y1 = std::get<1>(data1);
        
        utils::set_seed(123);
        auto mg2 = datasets::mackey_glass(500);
        auto data2 = datasets::to_forecasting(mg2);
        auto X2 = std::get<0>(data2);
        auto y2 = std::get<1>(data2);
        
        bool reproducible = true;
        for (int i = 0; i < std::min(static_cast<int>(X1.rows()), 10); ++i) {
            for (int j = 0; j < X1.cols(); ++j) {
                if (std::abs(X1(i, j) - X2(i, j)) > 1e-10f) {
                    reproducible = false;
                    break;
                }
            }
            if (!reproducible) break;
        }
        
        std::cout << "Reproducibility: " << (reproducible ? "✓ PASS" : "✗ FAIL") << std::endl;
        
        // Test error handling
        std::cout << "\n--- Error Handling Check ---" << std::endl;
        bool error_handling_works = true;
        
        try {
            Reservoir invalid_reservoir("test", -1);
            error_handling_works = false; // Should have thrown
        } catch (const std::invalid_argument&) {
            // Expected
        }
        
        try {
            matrix_generators::generate_internal_weights(10, -0.1f, 0.9f);
            error_handling_works = false; // Should have thrown
        } catch (const std::invalid_argument&) {
            // Expected
        }
        
        std::cout << "Error handling: " << (error_handling_works ? "✓ PASS" : "✗ FAIL") << std::endl;
        
        // ==================================================
        // INTEGRATION TESTING
        // ==================================================
        std::cout << "\n==================================================" << std::endl;
        std::cout << "  INTEGRATION TESTING" << std::endl;
        std::cout << "==================================================" << std::endl;
        
        std::cout << "Running full workflow integration test..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Complete workflow test
        utils::set_seed(42);
        auto mg_train = datasets::mackey_glass(1000);
        auto train_data = datasets::to_forecasting(mg_train);
        auto X_train = std::get<0>(train_data);
        auto y_train = std::get<1>(train_data);
        
        auto mg_test = datasets::mackey_glass(500);
        auto test_data = datasets::to_forecasting(mg_test);
        auto X_test = std::get<0>(test_data);
        auto y_test = std::get<1>(test_data);
        
        Reservoir reservoir("integration", 200);
        RidgeReadout readout("integration", 1);
        
        reservoir.initialize(&X_train);
        auto train_states = reservoir.forward(X_train);
        readout.fit(train_states, y_train);
        
        auto test_states = reservoir.forward(X_test);
        auto predictions = readout.forward(test_states);
        
        float mse = observables::mse(y_test, predictions);
        float rmse = observables::rmse(y_test, predictions);
        float r2 = observables::rsquare(y_test, predictions);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Integration test completed in: " << duration.count() << " ms" << std::endl;
        std::cout << "Performance metrics:" << std::endl;
        std::cout << "  MSE: " << std::fixed << std::setprecision(6) << mse << std::endl;
        std::cout << "  RMSE: " << rmse << std::endl;
        std::cout << "  R²: " << r2 << std::endl;
        
        // ==================================================
        // STAGE 7 SUMMARY
        // ==================================================
        std::cout << "\n==================================================" << std::endl;
        std::cout << "  STAGE 7 SUMMARY" << std::endl;
        std::cout << "==================================================" << std::endl;
        
        std::cout << "Stage 7 - Testing and Quality Assurance - COMPLETE!" << std::endl;
        std::cout << std::endl;
        
        std::cout << "✅ Quality Assurance Features Implemented:" << std::endl;
        std::cout << "   • Comprehensive performance benchmarking" << std::endl;
        std::cout << "   • Memory usage profiling" << std::endl;
        std::cout << "   • Robustness fuzz testing" << std::endl;
        std::cout << "   • Input validation testing" << std::endl;
        std::cout << "   • Numerical stability checks" << std::endl;
        std::cout << "   • Reproducibility validation" << std::endl;
        std::cout << "   • Error handling verification" << std::endl;
        std::cout << "   • Integration testing framework" << std::endl;
        std::cout << "   • CI/CD pipeline with multi-platform testing" << std::endl;
        std::cout << "   • Static analysis integration" << std::endl;
        std::cout << "   • Memory leak detection" << std::endl;
        std::cout << "   • Coverage analysis" << std::endl;
        std::cout << std::endl;
        
        std::cout << "🎯 Production-Ready Features:" << std::endl;
        std::cout << "   • Comprehensive test suite (70+ tests)" << std::endl;
        std::cout << "   • Multi-platform CI/CD (Ubuntu, Windows, macOS)" << std::endl;
        std::cout << "   • Multiple compiler support (GCC, Clang, MSVC)" << std::endl;
        std::cout << "   • Static analysis and quality checks" << std::endl;
        std::cout << "   • Performance benchmarking and profiling" << std::endl;
        std::cout << "   • Robust error handling and validation" << std::endl;
        std::cout << "   • Memory safety and leak detection" << std::endl;
        std::cout << std::endl;
        
        std::cout << "🚀 Ready for Stage 8:" << std::endl;
        std::cout << "   • Deployment and Packaging" << std::endl;
        std::cout << "   • Distribution preparation" << std::endl;
        std::cout << "   • Documentation finalization" << std::endl;
        std::cout << "   • Release preparation" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during Stage 7 tutorial: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}