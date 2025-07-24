#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

namespace reservoircpp {
namespace benchmark {

/**
 * @brief Simple benchmarking utility for performance testing
 */
class BenchmarkTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double, std::milli>;

    /**
     * @brief Benchmark result structure
     */
    struct Result {
        std::string name;
        double mean_ms;
        double min_ms;
        double max_ms;
        double std_dev_ms;
        size_t iterations;
    };

    /**
     * @brief Run a benchmark function multiple times and collect statistics
     * @param name Name of the benchmark
     * @param func Function to benchmark
     * @param iterations Number of iterations to run
     * @return Benchmark result with timing statistics
     */
    static Result benchmark(const std::string& name, 
                          std::function<void()> func, 
                          size_t iterations = 100) {
        std::vector<double> times;
        times.reserve(iterations);

        // Warm-up run
        func();

        // Actual benchmark runs
        for (size_t i = 0; i < iterations; ++i) {
            auto start = Clock::now();
            func();
            auto end = Clock::now();
            
            Duration elapsed = end - start;
            times.push_back(elapsed.count());
        }

        // Calculate statistics
        double sum = 0.0;
        double min_time = times[0];
        double max_time = times[0];
        
        for (double time : times) {
            sum += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        
        double mean = sum / iterations;
        
        // Calculate standard deviation
        double variance = 0.0;
        for (double time : times) {
            variance += (time - mean) * (time - mean);
        }
        variance /= iterations;
        double std_dev = std::sqrt(variance);

        return Result{name, mean, min_time, max_time, std_dev, iterations};
    }

    /**
     * @brief Print benchmark result in a formatted way
     */
    static void print_result(const Result& result) {
        std::cout << "Benchmark: " << result.name << std::endl;
        std::cout << "  Iterations: " << result.iterations << std::endl;
        std::cout << "  Mean time:  " << result.mean_ms << " ms" << std::endl;
        std::cout << "  Min time:   " << result.min_ms << " ms" << std::endl;
        std::cout << "  Max time:   " << result.max_ms << " ms" << std::endl;
        std::cout << "  Std dev:    " << result.std_dev_ms << " ms" << std::endl;
        std::cout << std::endl;
    }
};

/**
 * @brief Performance benchmark suite for reservoir computing operations
 */
class ReservoirBenchmarks {
public:
    /**
     * @brief Run all benchmarks and return results
     */
    static std::vector<BenchmarkTimer::Result> run_all_benchmarks();

    /**
     * @brief Benchmark matrix operations
     */
    static std::vector<BenchmarkTimer::Result> benchmark_matrix_operations();

    /**
     * @brief Benchmark activation functions
     */
    static std::vector<BenchmarkTimer::Result> benchmark_activations();

    /**
     * @brief Benchmark reservoir operations
     */
    static std::vector<BenchmarkTimer::Result> benchmark_reservoirs();

    /**
     * @brief Benchmark readout operations
     */
    static std::vector<BenchmarkTimer::Result> benchmark_readouts();

    /**
     * @brief Benchmark dataset generation
     */
    static std::vector<BenchmarkTimer::Result> benchmark_datasets();
};

/**
 * @brief Memory usage profiler
 */
class MemoryProfiler {
public:
    /**
     * @brief Get current memory usage in bytes
     */
    static size_t get_memory_usage();

    /**
     * @brief Profile memory usage of a function
     */
    static std::pair<size_t, size_t> profile_memory(std::function<void()> func);
};

} // namespace benchmark
} // namespace reservoircpp