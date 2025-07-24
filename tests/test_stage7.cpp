#include <catch2/catch_test_macros.hpp>
#include <reservoircpp/benchmark.hpp>
#include <reservoircpp/fuzz.hpp>
#include <reservoircpp/reservoircpp.hpp>
#include <iostream>

using namespace reservoircpp;

TEST_CASE("Stage 7 - Performance Benchmarks", "[stage7][performance]") {
    
    SECTION("Matrix Operations Benchmarks") {
        auto results = benchmark::ReservoirBenchmarks::benchmark_matrix_operations();
        
        REQUIRE(!results.empty());
        
        std::cout << "\n=== MATRIX OPERATIONS BENCHMARKS ===" << std::endl;
        for (const auto& result : results) {
            benchmark::BenchmarkTimer::print_result(result);
            
            // Basic sanity checks - operations should complete in reasonable time
            REQUIRE(result.mean_ms < 10000.0); // Less than 10 seconds
            REQUIRE(result.iterations > 0);
        }
    }
    
    SECTION("Activation Functions Benchmarks") {
        auto results = benchmark::ReservoirBenchmarks::benchmark_activations();
        
        REQUIRE(!results.empty());
        
        std::cout << "\n=== ACTIVATION FUNCTIONS BENCHMARKS ===" << std::endl;
        for (const auto& result : results) {
            benchmark::BenchmarkTimer::print_result(result);
            
            // Activation functions should be fast
            REQUIRE(result.mean_ms < 1000.0); // Less than 1 second
            REQUIRE(result.iterations > 0);
        }
    }
    
    SECTION("Reservoir Operations Benchmarks") {
        auto results = benchmark::ReservoirBenchmarks::benchmark_reservoirs();
        
        REQUIRE(!results.empty());
        
        std::cout << "\n=== RESERVOIR OPERATIONS BENCHMARKS ===" << std::endl;
        for (const auto& result : results) {
            benchmark::BenchmarkTimer::print_result(result);
            
            // Reservoir operations should complete in reasonable time
            REQUIRE(result.mean_ms < 30000.0); // Less than 30 seconds
            REQUIRE(result.iterations > 0);
        }
    }
    
    SECTION("Memory Usage Profiling") {
        auto [mem_before, mem_after] = benchmark::MemoryProfiler::profile_memory([]() {
            // Create some objects and measure memory usage
            Reservoir reservoir("memory_test", 500);
            Matrix input = Matrix::Random(1000, 10);
            reservoir.initialize(&input);
            auto states = reservoir.forward(input);
        });
        
        std::cout << "\nMemory usage - Before: " << mem_before 
                  << " bytes, After: " << mem_after << " bytes" << std::endl;
        
        // Memory usage should increase (on Linux systems where we can measure it)
        // On other systems, both values might be 0
        if (mem_before > 0 && mem_after > 0) {
            REQUIRE(mem_after >= mem_before);
        }
    }
}

TEST_CASE("Stage 7 - Fuzz Testing", "[stage7][fuzz]") {
    
    SECTION("Activation Functions Fuzzing") {
        fuzz::FuzzTester fuzzer(42); // Fixed seed for reproducibility
        auto results = fuzzer.fuzz_activations(100); // Reduced iterations for CI
        
        fuzz::FuzzTester::print_results(results);
        
        // All fuzz tests should pass
        for (const auto& result : results) {
            REQUIRE(result.passed);
            REQUIRE(result.iterations_completed > 0);
        }
    }
    
    SECTION("Matrix Generators Fuzzing") {
        fuzz::FuzzTester fuzzer(42);
        auto results = fuzzer.fuzz_matrix_generators(50);
        
        fuzz::FuzzTester::print_results(results);
        
        for (const auto& result : results) {
            REQUIRE(result.passed);
            REQUIRE(result.iterations_completed > 0);
        }
    }
    
    SECTION("Reservoir Operations Fuzzing") {
        fuzz::FuzzTester fuzzer(42);
        auto results = fuzzer.fuzz_reservoirs(50);
        
        fuzz::FuzzTester::print_results(results);
        
        for (const auto& result : results) {
            REQUIRE(result.passed);
            REQUIRE(result.iterations_completed > 0);
        }
    }
    
    SECTION("Readout Operations Fuzzing") {
        fuzz::FuzzTester fuzzer(42);
        auto results = fuzzer.fuzz_readouts(30);
        
        fuzz::FuzzTester::print_results(results);
        
        for (const auto& result : results) {
            REQUIRE(result.passed);
            REQUIRE(result.iterations_completed > 0);
        }
    }
    
    SECTION("Input Validation Fuzzing") {
        auto boundary_results = fuzz::InputValidationFuzzer::test_matrix_boundaries();
        auto param_results = fuzz::InputValidationFuzzer::test_parameter_ranges();
        auto memory_results = fuzz::InputValidationFuzzer::test_memory_limits();
        auto numerical_results = fuzz::InputValidationFuzzer::test_numerical_stability();
        
        std::cout << "\n=== INPUT VALIDATION FUZZ RESULTS ===" << std::endl;
        
        // Print all results
        std::vector<fuzz::FuzzTester::TestResult> all_validation_results;
        all_validation_results.insert(all_validation_results.end(), boundary_results.begin(), boundary_results.end());
        all_validation_results.insert(all_validation_results.end(), param_results.begin(), param_results.end());
        all_validation_results.insert(all_validation_results.end(), memory_results.begin(), memory_results.end());
        all_validation_results.insert(all_validation_results.end(), numerical_results.begin(), numerical_results.end());
        
        fuzz::FuzzTester::print_results(all_validation_results);
        
        // Most validation tests should pass (some may intentionally test failure cases)
        int passed_count = 0;
        for (const auto& result : all_validation_results) {
            if (result.passed) passed_count++;
        }
        
        // At least 50% should pass (allowing for some intentional failure tests)
        REQUIRE(passed_count >= static_cast<int>(all_validation_results.size() * 0.5));
    }
}

TEST_CASE("Stage 7 - Quality Assurance", "[stage7][quality]") {
    
    SECTION("Matrix Shape Validation") {
        // Test matrix shape consistency in operations
        Matrix a(10, 5);
        Matrix b(5, 8);
        Matrix c = a * b;
        
        REQUIRE(c.rows() == 10);
        REQUIRE(c.cols() == 8);
        
        // Test incompatible shapes
        Matrix d(3, 4);
        REQUIRE_THROWS([&]() {
            Matrix invalid = a * d; // Should throw due to incompatible dimensions
        }());
    }
    
    SECTION("Numerical Stability Checks") {
        // Test activation functions with extreme values
        Matrix extreme_positive(3, 3);
        extreme_positive.setConstant(1000.0f);
        
        Matrix extreme_negative(3, 3);
        extreme_negative.setConstant(-1000.0f);
        
        auto sigmoid_fn = activations::get_function("sigmoid");
        auto tanh_fn = activations::get_function("tanh");
        
        auto sigmoid_pos = sigmoid_fn(extreme_positive);
        auto sigmoid_neg = sigmoid_fn(extreme_negative);
        auto tanh_pos = tanh_fn(extreme_positive);
        auto tanh_neg = tanh_fn(extreme_negative);
        
        // Check for no NaN or infinity
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                REQUIRE(std::isfinite(sigmoid_pos(i, j)));
                REQUIRE(std::isfinite(sigmoid_neg(i, j)));
                REQUIRE(std::isfinite(tanh_pos(i, j)));
                REQUIRE(std::isfinite(tanh_neg(i, j)));
                
                // Check ranges
                REQUIRE(sigmoid_pos(i, j) >= 0.0f);
                REQUIRE(sigmoid_pos(i, j) <= 1.0f);
                REQUIRE(sigmoid_neg(i, j) >= 0.0f);
                REQUIRE(sigmoid_neg(i, j) <= 1.0f);
                REQUIRE(tanh_pos(i, j) >= -1.0f);
                REQUIRE(tanh_pos(i, j) <= 1.0f);
                REQUIRE(tanh_neg(i, j) >= -1.0f);
                REQUIRE(tanh_neg(i, j) <= 1.0f);
            }
        }
    }
    
    SECTION("Error Handling Validation") {
        // Test invalid reservoir creation
        REQUIRE_THROWS_AS(Reservoir("test", 0), std::invalid_argument);
        REQUIRE_THROWS_AS(Reservoir("test", -1), std::invalid_argument);
        
        // Test invalid matrix generator parameters
        REQUIRE_THROWS_AS(
            matrix_generators::generate_internal_weights(10, -0.1f, 0.9f),
            std::invalid_argument
        );
        
        REQUIRE_THROWS_AS(
            matrix_generators::generate_internal_weights(10, 1.1f, 0.9f),
            std::invalid_argument
        );
        
        // Test mismatched dimensions in readout training
        RidgeReadout readout("test", 5);
        Matrix states(100, 50);
        Matrix targets(90, 5); // Wrong number of samples
        
        REQUIRE_THROWS(readout.fit(states, targets));
    }
    
    SECTION("Memory Management Validation") {
        // Test that objects can be created and destroyed without issues
        std::vector<std::unique_ptr<Reservoir>> reservoirs;
        
        for (int i = 0; i < 10; ++i) {
            reservoirs.push_back(std::make_unique<Reservoir>("test_" + std::to_string(i), 100));
        }
        
        // Test that all reservoirs are functional
        Matrix input = Matrix::Random(50, 5);
        for (auto& reservoir : reservoirs) {
            reservoir->initialize(&input);
            auto states = reservoir->forward(input);
            REQUIRE(states.rows() == 50);
            REQUIRE(states.cols() == 100);
        }
        
        // Reservoirs will be automatically destroyed when going out of scope
    }
    
    SECTION("Reproducibility Validation") {
        // Test that operations with same seed produce same results
        utils::set_seed(42);
        Matrix random1 = utils::random_matrix(10, 10);
        
        utils::set_seed(42);
        Matrix random2 = utils::random_matrix(10, 10);
        
        // Should be identical
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 10; ++j) {
                REQUIRE(random1(i, j) == random2(i, j));
            }
        }
        
        // Test dataset reproducibility
        utils::set_seed(123);
        auto [X1, y1] = datasets::mackey_glass(1000);
        
        utils::set_seed(123);
        auto [X2, y2] = datasets::mackey_glass(1000);
        
        REQUIRE(X1.rows() == X2.rows());
        REQUIRE(X1.cols() == X2.cols());
        REQUIRE(y1.rows() == y2.rows());
        REQUIRE(y1.cols() == y2.cols());
        
        // Should be approximately equal (allowing for small floating point differences)
        for (int i = 0; i < X1.rows(); ++i) {
            for (int j = 0; j < X1.cols(); ++j) {
                REQUIRE(std::abs(X1(i, j) - X2(i, j)) < 1e-6f);
            }
        }
    }
}

TEST_CASE("Stage 7 - Integration and Regression Testing", "[stage7][integration]") {
    
    SECTION("Full Workflow Validation") {
        // Test complete reservoir computing workflow
        utils::set_seed(42);
        
        // Generate data
        auto [X_train, y_train] = datasets::mackey_glass(1000);
        auto [X_test, y_test] = datasets::mackey_glass(500);
        
        // Create and train model
        Reservoir reservoir("integration_test", 100);
        RidgeReadout readout("integration_test", 1);
        
        reservoir.initialize(&X_train);
        auto train_states = reservoir.forward(X_train);
        readout.fit(train_states, y_train);
        
        // Test predictions
        auto test_states = reservoir.forward(X_test);
        auto predictions = readout.forward(test_states);
        
        // Evaluate performance
        float mse = observables::mse(y_test, predictions);
        float rmse = observables::rmse(y_test, predictions);
        float r2 = observables::rsquare(y_test, predictions);
        
        // Performance should be reasonable
        REQUIRE(mse >= 0.0f);
        REQUIRE(rmse >= 0.0f);
        REQUIRE(r2 <= 1.0f);
        
        // For Mackey-Glass, we should get decent performance
        REQUIRE(mse < 1.0f); // Reasonable MSE
        REQUIRE(r2 > 0.5f);  // Reasonable R²
        
        std::cout << "\nIntegration Test Results:" << std::endl;
        std::cout << "MSE: " << mse << std::endl;
        std::cout << "RMSE: " << rmse << std::endl;
        std::cout << "R²: " << r2 << std::endl;
    }
    
    SECTION("Multi-Component Integration") {
        // Test interaction between different components
        utils::set_seed(123);
        
        // Test different reservoir types
        std::vector<std::unique_ptr<Node>> reservoirs;
        reservoirs.push_back(std::make_unique<Reservoir>("reservoir1", 50));
        reservoirs.push_back(std::make_unique<ESN>("esn1", 50));
        
        // Test different readouts
        std::vector<std::unique_ptr<RidgeReadout>> readouts;
        readouts.push_back(std::make_unique<RidgeReadout>("ridge1", 1));
        readouts.push_back(std::make_unique<RidgeReadout>("ridge2", 1));
        
        Matrix input = Matrix::Random(100, 5);
        Matrix targets = Matrix::Random(100, 1);
        
        // Test all combinations
        for (auto& reservoir : reservoirs) {
            reservoir->initialize(&input);
            auto states = reservoir->forward(input);
            
            for (auto& readout : readouts) {
                readout->fit(states, targets);
                auto predictions = readout->forward(states);
                
                REQUIRE(predictions.rows() == targets.rows());
                REQUIRE(predictions.cols() == targets.cols());
                
                // Should produce finite values
                for (int i = 0; i < predictions.rows(); ++i) {
                    for (int j = 0; j < predictions.cols(); ++j) {
                        REQUIRE(std::isfinite(predictions(i, j)));
                    }
                }
            }
        }
    }
    
    SECTION("Performance Regression Testing") {
        // Ensure performance hasn't regressed
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Standard benchmark task
        Reservoir reservoir("perf_test", 200);
        Matrix input = Matrix::Random(1000, 10);
        reservoir.initialize(&input);
        auto states = reservoir.forward(input);
        
        RidgeReadout readout("perf_test", 1);
        Matrix targets = Matrix::Random(1000, 1);
        readout.fit(states, targets);
        auto predictions = readout.forward(states);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\nPerformance test completed in: " << duration.count() << " ms" << std::endl;
        
        // Should complete in reasonable time (allowing for CI environment variations)
        REQUIRE(duration.count() < 10000); // Less than 10 seconds
        
        // Verify correctness
        REQUIRE(predictions.rows() == 1000);
        REQUIRE(predictions.cols() == 1);
        
        float mse = observables::mse(targets, predictions);
        REQUIRE(std::isfinite(mse));
        REQUIRE(mse >= 0.0f);
    }
}