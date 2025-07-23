/**
 * @file observables.hpp
 * @brief Metrics and observables for Reservoir Computing
 * 
 * This file contains the C++ port of the Python observables functionality,
 * implementing common metrics and reservoir-specific observables.
 */

#ifndef RESERVOIRCPP_OBSERVABLES_HPP
#define RESERVOIRCPP_OBSERVABLES_HPP

#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "reservoircpp/types.hpp"

namespace reservoircpp {
namespace observables {

/**
 * @brief Check that y_true and y_pred have the same shape
 * 
 * @param y_true True values
 * @param y_pred Predicted values
 * @throws std::invalid_argument if shapes don't match
 */
void check_arrays(const Matrix& y_true, const Matrix& y_pred);

/**
 * @brief Compute Mean Squared Error (MSE)
 * 
 * @param y_true True values
 * @param y_pred Predicted values
 * @return MSE value
 */
Float mse(const Matrix& y_true, const Matrix& y_pred);

/**
 * @brief Compute Root Mean Squared Error (RMSE)
 * 
 * @param y_true True values
 * @param y_pred Predicted values
 * @return RMSE value
 */
Float rmse(const Matrix& y_true, const Matrix& y_pred);

/**
 * @brief Compute Normalized Root Mean Squared Error (NRMSE)
 * 
 * @param y_true True values
 * @param y_pred Predicted values
 * @param normalization Normalization method ("var", "std", "range", "mean")
 * @return NRMSE value
 */
Float nrmse(const Matrix& y_true, const Matrix& y_pred, 
            const std::string& normalization = "var");

/**
 * @brief Compute R-squared (coefficient of determination)
 * 
 * @param y_true True values
 * @param y_pred Predicted values
 * @return R-squared value
 */
Float rsquare(const Matrix& y_true, const Matrix& y_pred);

/**
 * @brief Compute the spectral radius of a matrix
 * 
 * The spectral radius is the largest absolute eigenvalue.
 * 
 * @param W Input matrix
 * @param max_iter Maximum iterations for eigenvalue computation (unused for dense matrices)
 * @return Spectral radius
 */
Float spectral_radius(const Matrix& W, int max_iter = 1000);

/**
 * @brief Compute effective spectral radius for reservoir states
 * 
 * Estimates the spectral radius from reservoir state dynamics.
 * 
 * @param states Reservoir states over time (time x nodes)
 * @param n_samples Number of samples to use for estimation
 * @return Effective spectral radius
 */
Float effective_spectral_radius(const Matrix& states, int n_samples = 1000);

/**
 * @brief Compute memory capacity of a reservoir
 * 
 * Memory capacity measures how much information about past inputs
 * the reservoir can store and retrieve.
 * 
 * @param reservoir_states Reservoir states (time x nodes)
 * @param input_history Input history (time x input_dim)
 * @param max_delay Maximum delay to test
 * @return Memory capacity value
 */
Float memory_capacity(const Matrix& reservoir_states, 
                     const Matrix& input_history,
                     int max_delay = 100);

} // namespace observables
} // namespace reservoircpp

#endif // RESERVOIRCPP_OBSERVABLES_HPP