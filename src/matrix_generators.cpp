/**
 * @file matrix_generators.cpp
 * @brief Implementation of matrix generators for ReservoirCpp
 */

#include "reservoircpp/matrix_generators.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <random>

namespace reservoircpp {
namespace matrix_generators {

Matrix uniform(int rows, int cols, Float low, Float high, Float connectivity, int seed) {
    if (high <= low) {
        throw std::invalid_argument("'high' must be greater than 'low'");
    }
    
    Matrix matrix = detail::generate_random_values(rows, cols, "uniform", seed, low, high);
    
    if (connectivity < 1.0) {
        matrix = detail::apply_connectivity(matrix, connectivity, seed);
    }
    
    return matrix;
}

Matrix normal(int rows, int cols, Float mean, Float std, Float connectivity, int seed) {
    if (std <= 0.0) {
        throw std::invalid_argument("Standard deviation must be positive");
    }
    
    Matrix matrix = detail::generate_random_values(rows, cols, "normal", seed, mean, std);
    
    if (connectivity < 1.0) {
        matrix = detail::apply_connectivity(matrix, connectivity, seed);
    }
    
    return matrix;
}

Matrix bernoulli(int rows, int cols, Float prob, Float connectivity, int seed) {
    if (prob < 0.0 || prob > 1.0) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }
    
    Matrix matrix = detail::generate_random_values(rows, cols, "bernoulli", seed, prob, 0.0);
    
    if (connectivity < 1.0) {
        matrix = detail::apply_connectivity(matrix, connectivity, seed);
    }
    
    return matrix;
}

Matrix zeros(int rows, int cols) {
    return Matrix::Zero(rows, cols);
}

Matrix ones(int rows, int cols) {
    return Matrix::Ones(rows, cols);
}

SparseMatrix random_sparse(int rows, int cols, Float connectivity, 
                          const std::string& distribution, int seed) {
    if (connectivity <= 0.0 || connectivity > 1.0) {
        throw std::invalid_argument("Connectivity must be between 0 and 1");
    }
    
    // Create random number generator
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    std::uniform_real_distribution<Float> uniform_dist(0.0, 1.0);
    
    // Calculate number of non-zero elements
    int nnz = static_cast<int>(std::floor(connectivity * rows * cols));
    
    // Create triplet list for sparse matrix
    std::vector<Eigen::Triplet<Float>> triplets;
    triplets.reserve(nnz);
    
    // Generate random positions
    std::uniform_int_distribution<int> row_dist(0, rows - 1);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);
    
    for (int i = 0; i < nnz; ++i) {
        int row = row_dist(gen);
        int col = col_dist(gen);
        
        Float value = 0.0;
        if (distribution == "uniform") {
            std::uniform_real_distribution<Float> dist(-1.0, 1.0);
            value = dist(gen);
        } else if (distribution == "normal") {
            std::normal_distribution<Float> dist(0.0, 1.0);
            value = dist(gen);
        } else if (distribution == "bernoulli") {
            std::bernoulli_distribution dist(0.5);
            value = dist(gen) ? 1.0 : -1.0;
        } else {
            throw std::invalid_argument("Unknown distribution: " + distribution);
        }
        
        triplets.emplace_back(row, col, value);
    }
    
    SparseMatrix sparse_matrix(rows, cols);
    sparse_matrix.setFromTriplets(triplets.begin(), triplets.end());
    
    return sparse_matrix;
}

Matrix scale_spectral_radius(const Matrix& matrix, Float target_sr) {
    Float current_sr = spectral_radius(matrix);
    
    if (current_sr == 0.0) {
        return matrix; // Cannot scale zero matrix
    }
    
    Float scale_factor = target_sr / current_sr;
    return matrix * scale_factor;
}

Matrix generate_internal_weights(int units, Float connectivity, Float spectral_radius,
                                const std::string& distribution, int seed) {
    Matrix weights = uniform(units, units, -1.0, 1.0, connectivity, seed);
    
    // Apply spectral radius scaling
    weights = scale_spectral_radius(weights, spectral_radius);
    
    return weights;
}

Matrix generate_input_weights(int units, int input_dim, Float input_scaling,
                             Float connectivity, const std::string& distribution,
                             int seed) {
    Matrix weights;
    
    if (distribution == "uniform") {
        weights = uniform(units, input_dim, -1.0, 1.0, connectivity, seed);
    } else if (distribution == "normal") {
        weights = normal(units, input_dim, 0.0, 1.0, connectivity, seed);
    } else if (distribution == "bernoulli") {
        weights = bernoulli(units, input_dim, 0.5, connectivity, seed);
    } else {
        throw std::invalid_argument("Unknown distribution: " + distribution);
    }
    
    // Apply input scaling
    weights *= input_scaling;
    
    return weights;
}

Float spectral_radius(const Matrix& matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Matrix must be square to compute spectral radius");
    }
    
    if (matrix.rows() == 0) {
        return 0.0;
    }
    
    // Compute eigenvalues
    Eigen::EigenSolver<Matrix> solver(matrix);
    auto eigenvalues = solver.eigenvalues();
    
    // Find maximum absolute eigenvalue
    Float max_abs_eigenvalue = 0.0;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        Float abs_eigenvalue = std::abs(eigenvalues[i]);
        max_abs_eigenvalue = std::max(max_abs_eigenvalue, abs_eigenvalue);
    }
    
    return max_abs_eigenvalue;
}

namespace detail {

Matrix apply_connectivity(const Matrix& matrix, Float connectivity, int seed) {
    if (connectivity >= 1.0) {
        return matrix;
    }
    
    Matrix result = matrix;
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    std::uniform_real_distribution<Float> dist(0.0, 1.0);
    
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            if (dist(gen) >= connectivity) {
                result(i, j) = 0.0;
            }
        }
    }
    
    return result;
}

Matrix generate_random_values(int rows, int cols, const std::string& distribution,
                             int seed, Float param1, Float param2) {
    Matrix matrix(rows, cols);
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    
    if (distribution == "uniform") {
        std::uniform_real_distribution<Float> dist(param1, param2);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix(i, j) = dist(gen);
            }
        }
    } else if (distribution == "normal") {
        std::normal_distribution<Float> dist(param1, param2);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix(i, j) = dist(gen);
            }
        }
    } else if (distribution == "bernoulli") {
        std::bernoulli_distribution dist(param1);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix(i, j) = dist(gen) ? 1.0 : -1.0;
            }
        }
    } else {
        throw std::invalid_argument("Unknown distribution: " + distribution);
    }
    
    return matrix;
}

} // namespace detail

} // namespace matrix_generators
} // namespace reservoircpp