/**
 * @file debug_benchmark.cpp
 * @brief Debug benchmark hanging issue
 */

#include <iostream>
#include <chrono>
#include <reservoircpp/benchmark.hpp>
#include <reservoircpp/reservoir.hpp>

using namespace reservoircpp;

int main() {
    std::cout << "Testing individual benchmark components..." << std::endl;
    
    try {
        std::cout << "1. Testing matrix operations..." << std::endl;
        auto matrix_results = benchmark::ReservoirBenchmarks::benchmark_matrix_operations();
        std::cout << "Matrix operations completed: " << matrix_results.size() << " results" << std::endl;
        
        std::cout << "2. Testing activation functions..." << std::endl;
        auto activation_results = benchmark::ReservoirBenchmarks::benchmark_activations();
        std::cout << "Activation functions completed: " << activation_results.size() << " results" << std::endl;
        
        std::cout << "3. Testing reservoir operations..." << std::endl;
        auto reservoir_results = benchmark::ReservoirBenchmarks::benchmark_reservoirs();
        std::cout << "Reservoir operations completed: " << reservoir_results.size() << " results" << std::endl;
        
        std::cout << "4. Testing memory profiling..." << std::endl;
        auto [mem_before, mem_after] = benchmark::MemoryProfiler::profile_memory([]() {
            Reservoir reservoir("memory_test", 50);
            Matrix input = Matrix::Random(100, 10);
            reservoir.initialize(&input);
            auto states = reservoir.forward(input);
        });
        std::cout << "Memory profiling completed. Before: " << mem_before << ", After: " << mem_after << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "All benchmark components completed successfully!" << std::endl;
    return 0;
}
