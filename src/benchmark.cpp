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
    
    // Matrix multiplication benchmark
    auto matrix_mult = BenchmarkTimer::benchmark("Matrix Multiplication (100x100)", []() {
        Matrix a = Matrix::Random(100, 100);
        Matrix b = Matrix::Random(100, 100);
        Matrix c = a * b;
    }, 50);
    results.push_back(matrix_mult);
    
    // Matrix generation benchmark
    auto matrix_gen = BenchmarkTimer::benchmark("Matrix Generation (1000x1000)", []() {
        auto gen = reservoircpp::matrix_generators::generate_internal_weights(1000, 0.1f, 0.9f);
    }, 10);
    results.push_back(matrix_gen);
    
    // Eigenvalue computation benchmark
    auto eigenvals = BenchmarkTimer::benchmark("Eigenvalue Computation (50x50)", []() {
        Matrix m = Matrix::Random(50, 50);
        float sr = reservoircpp::observables::spectral_radius(m);
    }, 20);
    results.push_back(eigenvals);
    
    return results;
}

std::vector<BenchmarkTimer::Result> ReservoirBenchmarks::benchmark_activations() {
    std::vector<BenchmarkTimer::Result> results;
    
    Matrix input = Matrix::Random(1000, 100);
    
    // Sigmoid benchmark
    auto sigmoid = BenchmarkTimer::benchmark("Sigmoid Activation (1000x100)", [&input]() {
        auto sigmoid_fn = reservoircpp::activations::get_function("sigmoid");
        auto output = sigmoid_fn(input);
    }, 100);
    results.push_back(sigmoid);
    
    // Tanh benchmark
    auto tanh = BenchmarkTimer::benchmark("Tanh Activation (1000x100)", [&input]() {
        auto tanh_fn = reservoircpp::activations::get_function("tanh");
        auto output = tanh_fn(input);
    }, 100);
    results.push_back(tanh);
    
    // ReLU benchmark
    auto relu = BenchmarkTimer::benchmark("ReLU Activation (1000x100)", [&input]() {
        auto relu_fn = reservoircpp::activations::get_function("relu");
        auto output = relu_fn(input);
    }, 100);
    results.push_back(relu);
    
    return results;
}

std::vector<BenchmarkTimer::Result> ReservoirBenchmarks::benchmark_reservoirs() {
    std::vector<BenchmarkTimer::Result> results;
    
    // Reservoir creation benchmark
    auto reservoir_create = BenchmarkTimer::benchmark("Reservoir Creation (500 units)", []() {
        Reservoir reservoir("test", 500);
    }, 50);
    results.push_back(reservoir_create);
    
    // Reservoir forward pass benchmark
    Matrix input = Matrix::Random(1000, 10);
    auto reservoir_forward = BenchmarkTimer::benchmark("Reservoir Forward Pass (500 units, 1000 steps)", [&input]() {
        Reservoir reservoir("test", 500);
        reservoir.initialize(&input);
        auto states = reservoir.forward(input);
    }, 10);
    results.push_back(reservoir_forward);
    
    // ESN benchmark
    auto esn_forward = BenchmarkTimer::benchmark("ESN Forward Pass (200 units, 1000 steps)", [&input]() {
        ESN esn("test", 200);
        esn.initialize(&input);
        auto states = esn.forward(input);
    }, 10);
    results.push_back(esn_forward);
    
    return results;
}

std::vector<BenchmarkTimer::Result> ReservoirBenchmarks::benchmark_readouts() {
    std::vector<BenchmarkTimer::Result> results;
    
    Matrix states = Matrix::Random(1000, 100);
    Matrix targets = Matrix::Random(1000, 5);
    
    // Ridge readout training benchmark
    auto ridge_train = BenchmarkTimer::benchmark("Ridge Training (1000x100 -> 1000x5)", [&states, &targets]() {
        RidgeReadout readout("test", 5);
        readout.fit(states, targets);
    }, 20);
    results.push_back(ridge_train);
    
    // Ridge readout prediction benchmark
    RidgeReadout trained_readout("test", 5);
    trained_readout.fit(states, targets);
    auto ridge_predict = BenchmarkTimer::benchmark("Ridge Prediction (1000x100 -> 1000x5)", [&trained_readout, &states]() {
        auto output = trained_readout.forward(states);
    }, 100);
    results.push_back(ridge_predict);
    
    return results;
}

std::vector<BenchmarkTimer::Result> ReservoirBenchmarks::benchmark_datasets() {
    std::vector<BenchmarkTimer::Result> results;
    
    // Mackey-Glass generation benchmark
    auto mackey_glass = BenchmarkTimer::benchmark("Mackey-Glass Generation (5000 steps)", []() {
        auto data = reservoircpp::datasets::mackey_glass(5000);
    }, 10);
    results.push_back(mackey_glass);
    
    // Lorenz generation benchmark
    auto lorenz = BenchmarkTimer::benchmark("Lorenz Generation (5000 steps)", []() {
        auto data = reservoircpp::datasets::lorenz(5000);
    }, 10);
    results.push_back(lorenz);
    
    // NARMA generation benchmark
    auto narma = BenchmarkTimer::benchmark("NARMA Generation (5000 steps)", []() {
        auto data = reservoircpp::datasets::narma(5000);
        auto X = std::get<0>(data);
        auto y = std::get<1>(data);
    }, 10);
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