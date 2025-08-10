#include <reservoircpp/benchmark.hpp>
#include <reservoircpp/reservoircpp.hpp>
#include <fstream>
#include <cstring>
#include <cmath>

#ifdef __linux__
#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>
#endif

namespace reservoircpp {
namespace benchmark {

std::vector<BenchmarkTimer::Result> ReservoirBenchmarks::run_all_benchmarks() {
    std::vector<BenchmarkTimer::Result> all_results;
    
    auto matrix_results = benchmark_matrix_operations();
    auto activation_results = benchmark_activations();
    auto reservoir_results = benchmark_reservoirs();
    auto readout_results = benchmark_readouts();
    auto dataset_results = benchmark_datasets();
    
    all_results.insert(all_results.end(), matrix_results.begin(), matrix_results.end());
    all_results.insert(all_results.end(), activation_results.begin(), activation_results.end());
    all_results.insert(all_results.end(), reservoir_results.begin(), reservoir_results.end());
    all_results.insert(all_results.end(), readout_results.begin(), readout_results.end());
    all_results.insert(all_results.end(), dataset_results.begin(), dataset_results.end());
    
    return all_results;
}

std::vector<BenchmarkTimer::Result> ReservoirBenchmarks::benchmark_matrix_operations() {
    std::vector<BenchmarkTimer::Result> results;
    
    // Matrix multiplication benchmark (reduced size for testing)
    auto matrix_mult = BenchmarkTimer::benchmark("Matrix Multiplication (50x50)", []() {
        Matrix a = Matrix::Random(50, 50);
        Matrix b = Matrix::Random(50, 50);
        Matrix c = a * b;
    }, 5);
    results.push_back(matrix_mult);
    
    // Matrix generation benchmark (reduced size for testing)
    auto matrix_gen = BenchmarkTimer::benchmark("Matrix Generation (100x100)", []() {
        auto gen = reservoircpp::matrix_generators::generate_internal_weights(100, 0.1f, 0.9f);
    }, 3);
    results.push_back(matrix_gen);
    
    // Eigenvalue computation benchmark (reduced size for testing)
    auto eigenvals = BenchmarkTimer::benchmark("Eigenvalue Computation (20x20)", []() {
        Matrix m = Matrix::Random(20, 20);
        float sr = reservoircpp::observables::spectral_radius(m);
    }, 3);
    results.push_back(eigenvals);
    
    return results;
}

std::vector<BenchmarkTimer::Result> ReservoirBenchmarks::benchmark_activations() {
    std::vector<BenchmarkTimer::Result> results;
    
    Matrix input = Matrix::Random(100, 10); // Reduced size for testing
    
    // Sigmoid benchmark (reduced iterations)
    auto sigmoid = BenchmarkTimer::benchmark("Sigmoid Activation (100x10)", [&input]() {
        auto sigmoid_fn = reservoircpp::activations::get_function("sigmoid");
        auto output = sigmoid_fn(input);
    }, 10);
    results.push_back(sigmoid);
    
    // Tanh benchmark (reduced iterations)
    auto tanh = BenchmarkTimer::benchmark("Tanh Activation (100x10)", [&input]() {
        auto tanh_fn = reservoircpp::activations::get_function("tanh");
        auto output = tanh_fn(input);
    }, 10);
    results.push_back(tanh);
    
    // ReLU benchmark (reduced iterations)
    auto relu = BenchmarkTimer::benchmark("ReLU Activation (100x10)", [&input]() {
        auto relu_fn = reservoircpp::activations::get_function("relu");
        auto output = relu_fn(input);
    }, 10);
    results.push_back(relu);
    
    return results;
}

std::vector<BenchmarkTimer::Result> ReservoirBenchmarks::benchmark_reservoirs() {
    std::vector<BenchmarkTimer::Result> results;
    
    // Reservoir creation benchmark (reduced size)
    auto reservoir_create = BenchmarkTimer::benchmark("Reservoir Creation (50 units)", []() {
        Reservoir reservoir("test", 50);
    }, 5);
    results.push_back(reservoir_create);
    
    // Reservoir forward pass benchmark (much smaller for testing)
    Matrix input = Matrix::Random(100, 5);
    auto reservoir_forward = BenchmarkTimer::benchmark("Reservoir Forward Pass (50 units, 100 steps)", [&input]() {
        Reservoir reservoir("test", 50);
        reservoir.initialize(&input);
        auto states = reservoir.forward(input);
    }, 3);
    results.push_back(reservoir_forward);
    
    // ESN benchmark (much smaller for testing)
    auto esn_forward = BenchmarkTimer::benchmark("ESN Forward Pass (30 units, 100 steps)", [&input]() {
        ESN esn("test", 30);
        esn.initialize(&input);
        auto states = esn.forward(input);
    }, 3);
    results.push_back(esn_forward);
    
    return results;
}

std::vector<BenchmarkTimer::Result> ReservoirBenchmarks::benchmark_readouts() {
    std::vector<BenchmarkTimer::Result> results;
    
    Matrix states = Matrix::Random(100, 20); // Reduced size for testing
    Matrix targets = Matrix::Random(100, 3);
    
    // Ridge readout training benchmark (reduced size)
    auto ridge_train = BenchmarkTimer::benchmark("Ridge Training (100x20 -> 100x3)", [&states, &targets]() {
        RidgeReadout readout("test", 3);
        readout.fit(states, targets);
    }, 5);
    results.push_back(ridge_train);
    
    // Ridge readout prediction benchmark (reduced iterations)
    RidgeReadout trained_readout("test", 3);
    trained_readout.fit(states, targets);
    auto ridge_predict = BenchmarkTimer::benchmark("Ridge Prediction (100x20 -> 100x3)", [&trained_readout, &states]() {
        auto output = trained_readout.forward(states);
    }, 10);
    results.push_back(ridge_predict);
    
    return results;
}

std::vector<BenchmarkTimer::Result> ReservoirBenchmarks::benchmark_datasets() {
    std::vector<BenchmarkTimer::Result> results;
    
    // Mackey-Glass generation benchmark (reduced size)
    auto mackey_glass = BenchmarkTimer::benchmark("Mackey-Glass Generation (500 steps)", []() {
        auto data = reservoircpp::datasets::mackey_glass(500);
    }, 3);
    results.push_back(mackey_glass);
    
    // Lorenz generation benchmark (reduced size)
    auto lorenz = BenchmarkTimer::benchmark("Lorenz Generation (500 steps)", []() {
        auto data = reservoircpp::datasets::lorenz(500);
    }, 3);
    results.push_back(lorenz);
    
    // NARMA generation benchmark (reduced size)
    auto narma = BenchmarkTimer::benchmark("NARMA Generation (500 steps)", []() {
        auto data = reservoircpp::datasets::narma(500);
        auto X = std::get<0>(data);
        auto y = std::get<1>(data);
    }, 3);
    results.push_back(narma);
    
    return results;
}

size_t MemoryProfiler::get_memory_usage() {
#ifdef __linux__
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::string size_str = line.substr(7);
            size_str.erase(0, size_str.find_first_not_of(" \t"));
            size_str.erase(size_str.find_last_not_of(" \tkB") + 1);
            return std::stoull(size_str) * 1024; // Convert KB to bytes
        }
    }
#endif
    return 0; // Fallback for non-Linux systems
}

std::pair<size_t, size_t> MemoryProfiler::profile_memory(std::function<void()> func) {
    size_t mem_before = get_memory_usage();
    func();
    size_t mem_after = get_memory_usage();
    return {mem_before, mem_after};
}

} // namespace benchmark
} // namespace reservoircpp