/**
 * @file utils.hpp
 * @brief Utility functions for ReservoirCpp
 * 
 * This file contains utility functions for validation, random number generation,
 * and other common operations.
 */

#ifndef RESERVOIRCPP_UTILS_HPP
#define RESERVOIRCPP_UTILS_HPP

#include <Eigen/Dense>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
#include <mutex>
#include <algorithm>

#include "reservoircpp/types.hpp"

namespace reservoircpp {
namespace utils {

/**
 * @brief Random number generator singleton
 * 
 * Provides thread-safe random number generation with seeding capability
 */
class RandomGenerator {
public:
    static RandomGenerator& instance() {
        static RandomGenerator gen;
        return gen;
    }
    
    /**
     * @brief Set the random seed
     * 
     * @param seed Random seed value
     */
    void set_seed(unsigned int seed) {
        std::lock_guard<std::mutex> lock(mutex_);
        generator_.seed(seed);
    }
    
    /**
     * @brief Generate random float in [0, 1)
     * 
     * @return Random float value
     */
    Float uniform() {
        std::lock_guard<std::mutex> lock(mutex_);
        return uniform_dist_(generator_);
    }
    
    /**
     * @brief Generate random float in [min, max)
     * 
     * @param min Minimum value
     * @param max Maximum value
     * @return Random float value
     */
    Float uniform(Float min, Float max) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::uniform_real_distribution<Float> dist(min, max);
        return dist(generator_);
    }
    
    /**
     * @brief Generate random number from normal distribution
     * 
     * @param mean Mean value
     * @param std_dev Standard deviation
     * @return Random float value
     */
    Float normal(Float mean = 0.0, Float std_dev = 1.0) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::normal_distribution<Float> dist(mean, std_dev);
        return dist(generator_);
    }
    
    /**
     * @brief Generate random integer in [min, max]
     * 
     * @param min Minimum value (inclusive)
     * @param max Maximum value (inclusive)
     * @return Random integer value
     */
    int randint(int min, int max) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::uniform_int_distribution<int> dist(min, max);
        return dist(generator_);
    }
    
private:
    RandomGenerator() : generator_(std::random_device{}()), uniform_dist_(0.0, 1.0) {}
    
    std::mt19937 generator_;
    std::uniform_real_distribution<Float> uniform_dist_;
    std::mutex mutex_;
};

/**
 * @brief Generate random matrix with uniform distribution
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @param min Minimum value
 * @param max Maximum value
 * @return Random matrix
 */
inline Matrix random_uniform(int rows, int cols, Float min = 0.0, Float max = 1.0) {
    Matrix result(rows, cols);
    auto& rng = RandomGenerator::instance();
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = rng.uniform(min, max);
        }
    }
    
    return result;
}

/**
 * @brief Generate random matrix with normal distribution
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @param mean Mean value
 * @param std_dev Standard deviation
 * @return Random matrix
 */
inline Matrix random_normal(int rows, int cols, Float mean = 0.0, Float std_dev = 1.0) {
    Matrix result(rows, cols);
    auto& rng = RandomGenerator::instance();
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = rng.normal(mean, std_dev);
        }
    }
    
    return result;
}

/**
 * @brief Set global random seed
 * 
 * @param seed Random seed value
 */
inline void set_seed(unsigned int seed) {
    RandomGenerator::instance().set_seed(seed);
}

/**
 * @brief Validation utilities
 */
namespace validation {

/**
 * @brief Check if matrix has expected dimensions
 * 
 * @param matrix Matrix to check
 * @param expected_rows Expected number of rows (-1 to ignore)
 * @param expected_cols Expected number of columns (-1 to ignore)
 * @param name Name for error messages
 * @throws std::invalid_argument if dimensions don't match
 */
inline void check_dimensions(const Matrix& matrix, int expected_rows, int expected_cols, 
                           const std::string& name = "matrix") {
    if (expected_rows >= 0 && matrix.rows() != expected_rows) {
        std::ostringstream msg;
        msg << name << " should have " << expected_rows << " rows, got " << matrix.rows();
        throw std::invalid_argument(msg.str());
    }
    
    if (expected_cols >= 0 && matrix.cols() != expected_cols) {
        std::ostringstream msg;
        msg << name << " should have " << expected_cols << " columns, got " << matrix.cols();
        throw std::invalid_argument(msg.str());
    }
}

/**
 * @brief Check if matrix is not empty
 * 
 * @param matrix Matrix to check
 * @param name Name for error messages
 * @throws std::invalid_argument if matrix is empty
 */
inline void check_not_empty(const Matrix& matrix, const std::string& name = "matrix") {
    if (matrix.rows() == 0 || matrix.cols() == 0) {
        throw std::invalid_argument(name + " cannot be empty");
    }
}

/**
 * @brief Check if vector has expected size
 * 
 * @param vector Vector to check
 * @param expected_size Expected size (-1 to ignore)
 * @param name Name for error messages
 * @throws std::invalid_argument if size doesn't match
 */
inline void check_vector_size(const Vector& vector, int expected_size, 
                            const std::string& name = "vector") {
    if (expected_size >= 0 && vector.size() != expected_size) {
        std::ostringstream msg;
        msg << name << " should have size " << expected_size << ", got " << vector.size();
        throw std::invalid_argument(msg.str());
    }
}

/**
 * @brief Check if matrices have compatible dimensions for multiplication
 * 
 * @param a First matrix
 * @param b Second matrix
 * @param name_a Name of first matrix
 * @param name_b Name of second matrix
 * @throws std::invalid_argument if dimensions are incompatible
 */
inline void check_multiplication_compatible(const Matrix& a, const Matrix& b,
                                          const std::string& name_a = "matrix A",
                                          const std::string& name_b = "matrix B") {
    if (a.cols() != b.rows()) {
        std::ostringstream msg;
        msg << name_a << " columns (" << a.cols() << ") must match " 
            << name_b << " rows (" << b.rows() << ") for multiplication";
        throw std::invalid_argument(msg.str());
    }
}

} // namespace validation

/**
 * @brief Array utilities
 */
namespace array {

/**
 * @brief Convert Shape to string representation
 * 
 * @param shape Shape vector
 * @return String representation
 */
inline std::string shape_to_string(const Shape& shape) {
    std::ostringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape[i];
    }
    ss << ")";
    return ss.str();
}

/**
 * @brief Get shape of matrix
 * 
 * @param matrix Input matrix
 * @return Shape vector
 */
inline Shape get_shape(const Matrix& matrix) {
    return {static_cast<int>(matrix.rows()), static_cast<int>(matrix.cols())};
}

/**
 * @brief Check if two shapes are equal
 * 
 * @param shape1 First shape
 * @param shape2 Second shape
 * @return true if shapes are equal
 */
inline bool shapes_equal(const Shape& shape1, const Shape& shape2) {
    return shape1.size() == shape2.size() && 
           std::equal(shape1.begin(), shape1.end(), shape2.begin());
}

} // namespace array

} // namespace utils
} // namespace reservoircpp

#endif // RESERVOIRCPP_UTILS_HPP