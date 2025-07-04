/**
 * @file matrix_generators.hpp
 * @brief Matrix generators for ReservoirCpp
 * 
 * This file contains the C++ port of the Python mat_gen.py,
 * providing weight matrix initialization utilities for reservoir computing.
 */

#ifndef RESERVOIRCPP_MATRIX_GENERATORS_HPP
#define RESERVOIRCPP_MATRIX_GENERATORS_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <functional>
#include <random>
#include <string>
#include <stdexcept>
#include <cmath>

#include "reservoircpp/types.hpp"
#include "reservoircpp/utils.hpp"

namespace reservoircpp {
namespace matrix_generators {

/**
 * @brief Matrix generator function type
 */
using MatrixGenerator = std::function<Matrix(int, int)>;

/**
 * @brief Sparse matrix generator function type
 */
using SparseMatrixGenerator = std::function<SparseMatrix(int, int)>;

/**
 * @brief Generate a matrix with uniform random values
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @param low Lower bound of uniform distribution (default: -1.0)
 * @param high Upper bound of uniform distribution (default: 1.0)
 * @param connectivity Connectivity ratio (default: 1.0 for dense)
 * @param seed Random seed (default: use global seed)
 * @return Matrix with uniform random values
 */
Matrix uniform(int rows, int cols, Float low = -1.0, Float high = 1.0, 
               Float connectivity = 1.0, int seed = -1);

/**
 * @brief Generate a matrix with normal random values
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @param mean Mean of normal distribution (default: 0.0)
 * @param std Standard deviation of normal distribution (default: 1.0)
 * @param connectivity Connectivity ratio (default: 1.0 for dense)
 * @param seed Random seed (default: use global seed)
 * @return Matrix with normal random values
 */
Matrix normal(int rows, int cols, Float mean = 0.0, Float std = 1.0, 
              Float connectivity = 1.0, int seed = -1);

/**
 * @brief Generate a matrix with Bernoulli random values
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @param prob Probability of 1 (default: 0.5)
 * @param connectivity Connectivity ratio (default: 1.0 for dense)
 * @param seed Random seed (default: use global seed)
 * @return Matrix with Bernoulli random values (-1 or 1)
 */
Matrix bernoulli(int rows, int cols, Float prob = 0.5, 
                 Float connectivity = 1.0, int seed = -1);

/**
 * @brief Generate a matrix filled with zeros
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Matrix filled with zeros
 */
Matrix zeros(int rows, int cols);

/**
 * @brief Generate a matrix filled with ones
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Matrix filled with ones
 */
Matrix ones(int rows, int cols);

/**
 * @brief Generate random sparse matrix
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @param connectivity Connectivity ratio (0.0 to 1.0)
 * @param distribution Distribution type ("uniform", "normal", "bernoulli")
 * @param seed Random seed (default: use global seed)
 * @return Sparse matrix
 */
SparseMatrix random_sparse(int rows, int cols, Float connectivity = 0.1,
                          const std::string& distribution = "uniform", 
                          int seed = -1);

/**
 * @brief Scale matrix to target spectral radius
 * 
 * @param matrix Input matrix
 * @param target_sr Target spectral radius
 * @return Scaled matrix
 */
Matrix scale_spectral_radius(const Matrix& matrix, Float target_sr);

/**
 * @brief Generate internal weights for reservoir
 * 
 * @param units Number of reservoir units
 * @param connectivity Connectivity ratio (default: 0.1)
 * @param spectral_radius Target spectral radius (default: 0.9)
 * @param distribution Distribution type (default: "uniform")
 * @param seed Random seed (default: use global seed)
 * @return Internal weight matrix
 */
Matrix generate_internal_weights(int units, Float connectivity = 0.1,
                                Float spectral_radius = 0.9,
                                const std::string& distribution = "uniform",
                                int seed = -1);

/**
 * @brief Generate input weights for reservoir
 * 
 * @param units Number of reservoir units
 * @param input_dim Input dimension
 * @param input_scaling Input scaling factor (default: 1.0)
 * @param connectivity Connectivity ratio (default: 1.0)
 * @param distribution Distribution type (default: "uniform")
 * @param seed Random seed (default: use global seed)
 * @return Input weight matrix
 */
Matrix generate_input_weights(int units, int input_dim, Float input_scaling = 1.0,
                             Float connectivity = 1.0,
                             const std::string& distribution = "uniform",
                             int seed = -1);

/**
 * @brief Compute spectral radius of a matrix
 * 
 * @param matrix Input matrix
 * @return Spectral radius (largest eigenvalue magnitude)
 */
Float spectral_radius(const Matrix& matrix);

namespace detail {

/**
 * @brief Apply connectivity mask to matrix
 * 
 * @param matrix Input matrix
 * @param connectivity Connectivity ratio (0.0 to 1.0)
 * @param seed Random seed
 * @return Matrix with connectivity mask applied
 */
Matrix apply_connectivity(const Matrix& matrix, Float connectivity, int seed);

/**
 * @brief Generate random values from distribution
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @param distribution Distribution type
 * @param seed Random seed
 * @param param1 First distribution parameter
 * @param param2 Second distribution parameter
 * @return Matrix with random values
 */
Matrix generate_random_values(int rows, int cols, const std::string& distribution,
                             int seed, Float param1 = 0.0, Float param2 = 1.0);

} // namespace detail

} // namespace matrix_generators
} // namespace reservoircpp

#endif // RESERVOIRCPP_MATRIX_GENERATORS_HPP