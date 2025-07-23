/**
 * @file gpu.hpp
 * @brief GPU support framework for ReservoirCpp
 * 
 * This module provides GPU acceleration support for reservoir computing
 * operations using CUDA or other GPU computing frameworks.
 * 
 * @author ReservoirCpp Development Team
 * @date 2024
 * @license MIT License
 */

#ifndef RESERVOIRCPP_GPU_HPP
#define RESERVOIRCPP_GPU_HPP

#include "reservoircpp/types.hpp"
#include "reservoircpp/node.hpp"
#include "reservoircpp/reservoir.hpp"
#include "reservoircpp/readout.hpp"
#include <memory>
#include <string>

namespace reservoircpp {
namespace gpu {

/**
 * @brief GPU device information
 */
struct DeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    
    DeviceInfo() : device_id(-1), total_memory(0), free_memory(0),
                  compute_capability_major(0), compute_capability_minor(0),
                  multiprocessor_count(0) {}
};

/**
 * @brief GPU memory management
 */
class GpuMemoryManager {
public:
    /**
     * @brief Get GPU device information
     * @param device_id GPU device ID
     * @return Device information
     */
    static DeviceInfo get_device_info(int device_id = 0);

    /**
     * @brief Get number of available GPU devices
     * @return Number of devices
     */
    static int get_device_count();

    /**
     * @brief Set active GPU device
     * @param device_id Device ID to activate
     * @return True if successful
     */
    static bool set_device(int device_id);

    /**
     * @brief Check if GPU is available
     * @return True if GPU is available
     */
    static bool is_gpu_available();

    /**
     * @brief Get memory usage for current device
     * @return Pair of (used_memory, total_memory) in bytes
     */
    static std::pair<size_t, size_t> get_memory_usage();

    /**
     * @brief Synchronize GPU operations
     */
    static void synchronize();
};

/**
 * @brief GPU matrix operations
 */
class GpuMatrix {
public:
    /**
     * @brief Constructor
     * @param rows Number of rows
     * @param cols Number of columns
     */
    GpuMatrix(size_t rows = 0, size_t cols = 0);

    /**
     * @brief Constructor from CPU matrix
     * @param cpu_matrix CPU matrix to copy
     */
    explicit GpuMatrix(const Matrix& cpu_matrix);

    /**
     * @brief Destructor
     */
    ~GpuMatrix();

    // Copy constructor and assignment operator
    GpuMatrix(const GpuMatrix& other);
    GpuMatrix& operator=(const GpuMatrix& other);

    // Move constructor and assignment operator
    GpuMatrix(GpuMatrix&& other) noexcept;
    GpuMatrix& operator=(GpuMatrix&& other) noexcept;

    /**
     * @brief Get matrix dimensions
     */
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }

    /**
     * @brief Copy data from CPU to GPU
     * @param cpu_matrix CPU matrix
     */
    void copy_from_cpu(const Matrix& cpu_matrix);

    /**
     * @brief Copy data from GPU to CPU
     * @return CPU matrix
     */
    Matrix copy_to_cpu() const;

    /**
     * @brief Resize matrix
     * @param rows New number of rows
     * @param cols New number of columns
     */
    void resize(size_t rows, size_t cols);

    /**
     * @brief Fill matrix with value
     * @param value Fill value
     */
    void fill(float value);

    /**
     * @brief Matrix multiplication: C = A * B
     * @param A First matrix
     * @param B Second matrix
     * @param C Result matrix
     */
    static void multiply(const GpuMatrix& A, const GpuMatrix& B, GpuMatrix& C);

    /**
     * @brief Matrix addition: C = A + B
     * @param A First matrix
     * @param B Second matrix
     * @param C Result matrix
     */
    static void add(const GpuMatrix& A, const GpuMatrix& B, GpuMatrix& C);

    /**
     * @brief Element-wise multiplication: C = A .* B
     * @param A First matrix
     * @param B Second matrix
     * @param C Result matrix
     */
    static void element_multiply(const GpuMatrix& A, const GpuMatrix& B, GpuMatrix& C);

    /**
     * @brief Apply activation function element-wise
     * @param input Input matrix
     * @param output Output matrix
     * @param activation_name Name of activation function
     */
    static void apply_activation(const GpuMatrix& input, GpuMatrix& output,
                               const std::string& activation_name);

    /**
     * @brief Generate random matrix
     * @param matrix Matrix to fill
     * @param distribution Distribution type ("uniform", "normal")
     * @param param1 First parameter (min for uniform, mean for normal)
     * @param param2 Second parameter (max for uniform, std for normal)
     */
    static void random_fill(GpuMatrix& matrix, const std::string& distribution,
                           float param1, float param2);

private:
    void* data_;  ///< GPU memory pointer
    size_t rows_;
    size_t cols_;

    void allocate_memory();
    void deallocate_memory();
};

/**
 * @brief GPU-accelerated reservoir
 */
class GpuReservoir : public Node {
public:
    /**
     * @brief Constructor
     * @param name Node name
     * @param units Number of reservoir units
     * @param input_scaling Input scaling factor
     * @param spectral_radius Spectral radius of reservoir matrix
     * @param leak_rate Leak rate (0-1)
     * @param connectivity Sparsity of reservoir connections (0-1)
     * @param input_connectivity Sparsity of input connections (0-1)
     * @param seed Random seed
     */
    GpuReservoir(const std::string& name = "GpuReservoir",
                 size_t units = 500,
                 float input_scaling = 1.0f,
                 float spectral_radius = 0.95f,
                 float leak_rate = 1.0f,
                 float connectivity = 0.1f,
                 float input_connectivity = 1.0f,
                 unsigned int seed = 42);

    Matrix forward(const Matrix& input) override;
    void initialize() override;
    void reset_state() override;

    /**
     * @brief Forward pass using GPU matrices
     * @param gpu_input GPU input matrix
     * @param gpu_output GPU output matrix
     */
    void forward_gpu(const GpuMatrix& gpu_input, GpuMatrix& gpu_output);

    // Getters
    size_t get_units() const { return units_; }
    float get_input_scaling() const { return input_scaling_; }
    float get_spectral_radius() const { return spectral_radius_; }
    float get_leak_rate() const { return leak_rate_; }

private:
    size_t units_;
    float input_scaling_;
    float spectral_radius_;
    float leak_rate_;
    float connectivity_;
    float input_connectivity_;
    unsigned int seed_;

    GpuMatrix W_;           ///< Reservoir weight matrix
    GpuMatrix Win_;         ///< Input weight matrix
    GpuMatrix state_;       ///< Current reservoir state
    GpuMatrix temp_state_;  ///< Temporary state for computations
};

/**
 * @brief GPU-accelerated readout
 */
class GpuReadout : public Node {
public:
    /**
     * @brief Constructor
     * @param name Node name
     * @param output_dim Output dimensionality
     * @param ridge Ridge regularization parameter
     */
    GpuReadout(const std::string& name = "GpuReadout",
               size_t output_dim = 1,
               float ridge = 1e-6f);

    Matrix forward(const Matrix& input) override;
    void initialize() override;
    void reset_state() override;

    /**
     * @brief Fit readout weights using GPU
     * @param X Input data (CPU matrix)
     * @param y Target data (CPU matrix)
     */
    void fit(const Matrix& X, const Matrix& y);

    /**
     * @brief Forward pass using GPU matrices
     * @param gpu_input GPU input matrix
     * @param gpu_output GPU output matrix
     */
    void forward_gpu(const GpuMatrix& gpu_input, GpuMatrix& gpu_output);

    // Getters
    size_t get_output_dim() const { return output_dim_; }
    float get_ridge() const { return ridge_; }
    Matrix get_weights() const;

private:
    size_t output_dim_;
    float ridge_;

    GpuMatrix Wout_;  ///< Output weight matrix
    GpuMatrix bias_;  ///< Bias vector
    bool fitted_ = false;
};

/**
 * @brief GPU acceleration utilities
 */
class GpuUtils {
public:
    /**
     * @brief Initialize GPU context
     * @param device_id GPU device ID to use
     * @return True if successful
     */
    static bool initialize(int device_id = 0);

    /**
     * @brief Cleanup GPU context
     */
    static void cleanup();

    /**
     * @brief Check if operations should use GPU
     * @param matrix_size Size of matrices involved
     * @return True if GPU should be used
     */
    static bool should_use_gpu(size_t matrix_size);

    /**
     * @brief Benchmark GPU vs CPU performance
     * @param matrix_size Size of test matrices
     * @param n_iterations Number of test iterations
     * @return Ratio of GPU time to CPU time
     */
    static float benchmark_performance(size_t matrix_size, int n_iterations = 10);

    /**
     * @brief Auto-detect optimal batch size for GPU
     * @param input_size Size of input data
     * @param available_memory Available GPU memory
     * @return Optimal batch size
     */
    static size_t auto_batch_size(size_t input_size, size_t available_memory);

    /**
     * @brief Convert CPU node to GPU equivalent
     * @param cpu_node CPU node to convert
     * @return GPU node or nullptr if not supported
     */
    static std::unique_ptr<Node> convert_to_gpu(const Node& cpu_node);

private:
    static bool initialized_;
    static int current_device_;
};

/**
 * @brief CUDA error checking utilities
 */
class CudaError {
public:
    /**
     * @brief Check CUDA error and throw exception if needed
     * @param cuda_call CUDA function call result
     * @param file Source file name
     * @param line Source line number
     */
    static void check(void* cuda_call, const char* file, int line);

    /**
     * @brief Get error string for CUDA error code
     * @param error CUDA error code
     * @return Error description
     */
    static std::string get_error_string(int error);
};

// Macro for CUDA error checking
#define CUDA_CHECK(call) reservoircpp::gpu::CudaError::check((call), __FILE__, __LINE__)

/**
 * @brief GPU memory pool for efficient allocation
 */
class GpuMemoryPool {
public:
    /**
     * @brief Get singleton instance
     */
    static GpuMemoryPool& instance();

    /**
     * @brief Allocate GPU memory
     * @param size Size in bytes
     * @return GPU memory pointer
     */
    void* allocate(size_t size);

    /**
     * @brief Deallocate GPU memory
     * @param ptr GPU memory pointer
     * @param size Size in bytes
     */
    void deallocate(void* ptr, size_t size);

    /**
     * @brief Get memory usage statistics
     * @return Pair of (allocated, peak_allocated) in bytes
     */
    std::pair<size_t, size_t> get_stats() const;

    /**
     * @brief Clear all cached memory
     */
    void clear_cache();

private:
    GpuMemoryPool() = default;
    
    size_t allocated_bytes_ = 0;
    size_t peak_allocated_bytes_ = 0;
    std::vector<std::pair<void*, size_t>> free_blocks_;
};

} // namespace gpu
} // namespace reservoircpp

#endif // RESERVOIRCPP_GPU_HPP