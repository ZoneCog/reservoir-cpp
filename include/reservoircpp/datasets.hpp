/**
 * @file datasets.hpp
 * @brief Dataset generators for Reservoir Computing
 * 
 * This file contains the C++ port of the Python datasets functionality,
 * implementing chaotic time series generators and data preprocessing utilities.
 */

#ifndef RESERVOIRCPP_DATASETS_HPP
#define RESERVOIRCPP_DATASETS_HPP

#include <vector>
#include <tuple>
#include <string>

#include "reservoircpp/types.hpp"

namespace reservoircpp {
namespace datasets {

/**
 * @brief Generate Mackey-Glass time series
 * 
 * The Mackey-Glass equation is a time-delay differential equation
 * commonly used as a benchmark for reservoir computing.
 * 
 * @param n_timesteps Number of time steps to generate
 * @param tau Time delay parameter (default: 17)
 * @param a Parameter a (default: 0.2)
 * @param b Parameter b (default: 0.1) 
 * @param n Parameter n (default: 10)
 * @param h Step size for integration (default: 1.0)
 * @param x0 Initial condition (default: 1.2)
 * @param washout Number of initial steps to discard (default: 100)
 * @return Matrix of shape (n_timesteps, 1)
 */
Matrix mackey_glass(int n_timesteps, int tau = 17, Float a = 0.2, Float b = 0.1,
                   Float n = 10.0, Float h = 1.0, Float x0 = 1.2, int washout = 100);

/**
 * @brief Generate Lorenz system time series
 * 
 * The Lorenz system is a chaotic dynamical system that exhibits
 * sensitive dependence on initial conditions.
 * 
 * @param n_timesteps Number of time steps to generate
 * @param dt Time step for integration (default: 0.01)
 * @param sigma Parameter sigma (default: 10.0)
 * @param rho Parameter rho (default: 28.0)
 * @param beta Parameter beta (default: 8.0/3.0)
 * @param x0 Initial condition for x (default: 1.0)
 * @param y0 Initial condition for y (default: 1.0)
 * @param z0 Initial condition for z (default: 1.0)
 * @param washout Number of initial steps to discard (default: 100)
 * @return Matrix of shape (n_timesteps, 3)
 */
Matrix lorenz(int n_timesteps, Float dt = 0.01, Float sigma = 10.0, 
              Float rho = 28.0, Float beta = 8.0/3.0,
              Float x0 = 1.0, Float y0 = 1.0, Float z0 = 1.0, int washout = 100);

/**
 * @brief Generate Hénon map time series
 * 
 * The Hénon map is a discrete-time chaotic map.
 * 
 * @param n_timesteps Number of time steps to generate
 * @param a Parameter a (default: 1.4)
 * @param b Parameter b (default: 0.3)
 * @param x0 Initial condition for x (default: 0.0)
 * @param y0 Initial condition for y (default: 0.0)
 * @param washout Number of initial steps to discard (default: 100)
 * @return Matrix of shape (n_timesteps, 2)
 */
Matrix henon_map(int n_timesteps, Float a = 1.4, Float b = 0.3,
                Float x0 = 0.0, Float y0 = 0.0, int washout = 100);

/**
 * @brief Generate logistic map time series
 * 
 * The logistic map is a simple discrete-time chaotic map.
 * 
 * @param n_timesteps Number of time steps to generate
 * @param r Parameter r (default: 4.0 for chaotic behavior)
 * @param x0 Initial condition (default: 0.5)
 * @param washout Number of initial steps to discard (default: 100)
 * @return Matrix of shape (n_timesteps, 1)
 */
Matrix logistic_map(int n_timesteps, Float r = 4.0, Float x0 = 0.5, int washout = 100);

/**
 * @brief Generate NARMA (Nonlinear AutoRegressive Moving Average) time series
 * 
 * NARMA is a commonly used benchmark for reservoir computing.
 * 
 * @param n_timesteps Number of time steps to generate
 * @param order NARMA order (default: 10)
 * @param alpha Parameter alpha (default: 0.3)
 * @param beta Parameter beta (default: 0.05)
 * @param gamma Parameter gamma (default: 1.5)
 * @param delta Parameter delta (default: 0.1)
 * @param washout Number of initial steps to discard (default: 100)
 * @return Tuple of (input, target) matrices
 */
std::tuple<Matrix, Matrix> narma(int n_timesteps, int order = 10, 
                                Float alpha = 0.3, Float beta = 0.05, 
                                Float gamma = 1.5, Float delta = 0.1, 
                                int washout = 100);

/**
 * @brief Split a time series for forecasting tasks
 * 
 * Transform a time series X into input values X_t and output values X_{t+forecast}.
 * 
 * @param timeseries Input time series
 * @param forecast Number of time steps to forecast ahead (default: 1)
 * @param test_size Size of test set (default: 0, no split)
 * @return Tuple of (X_train, y_train) or (X_train, X_test, y_train, y_test)
 */
std::tuple<Matrix, Matrix> to_forecasting(const Matrix& timeseries, int forecast = 1);
std::tuple<Matrix, Matrix, Matrix, Matrix> to_forecasting_with_split(
    const Matrix& timeseries, int forecast = 1, int test_size = 0);

/**
 * @brief One-hot encode categorical data
 * 
 * @param labels Integer labels to encode
 * @param num_classes Number of classes (if 0, inferred from data)
 * @return One-hot encoded matrix
 */
Matrix one_hot_encode(const std::vector<int>& labels, int num_classes = 0);

/**
 * @brief Multiple superimposed oscillators (MSO) task
 * 
 * This task is used to evaluate reservoir resistance to perturbations.
 * 
 * @param n_timesteps Number of time steps to generate
 * @param frequencies List of frequencies for the oscillators
 * @param normalize Whether to normalize to [-1, 1] range (default: true)
 * @return Matrix of shape (n_timesteps, 1)
 */
Matrix mso(int n_timesteps, const std::vector<Float>& frequencies, bool normalize = true);

/**
 * @brief MSO with 2 frequencies (0.2, 0.311)
 * 
 * @param n_timesteps Number of time steps to generate
 * @param normalize Whether to normalize to [-1, 1] range (default: true)
 * @return Matrix of shape (n_timesteps, 1)
 */
Matrix mso2(int n_timesteps, bool normalize = true);

/**
 * @brief MSO with 8 frequencies
 * 
 * @param n_timesteps Number of time steps to generate
 * @param normalize Whether to normalize to [-1, 1] range (default: true)
 * @return Matrix of shape (n_timesteps, 1)
 */
Matrix mso8(int n_timesteps, bool normalize = true);

} // namespace datasets
} // namespace reservoircpp

#endif // RESERVOIRCPP_DATASETS_HPP