/**
 * @file observables.cpp
 * @brief Implementation of observables for Reservoir Computing
 */

#include "reservoircpp/observables.hpp"
#include <algorithm>
#include <numeric>

namespace reservoircpp {
namespace observables {

void check_arrays(const Matrix& y_true, const Matrix& y_pred) {
    if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols()) {
        throw std::invalid_argument(
            "Shape mismatch between y_true and y_pred: (" + 
            std::to_string(y_true.rows()) + "x" + std::to_string(y_true.cols()) + 
            ") vs (" + std::to_string(y_pred.rows()) + "x" + std::to_string(y_pred.cols()) + ")"
        );
    }
}

Float mse(const Matrix& y_true, const Matrix& y_pred) {
    check_arrays(y_true, y_pred);
    
    Matrix diff = y_true - y_pred;
    return diff.array().square().mean();
}

Float rmse(const Matrix& y_true, const Matrix& y_pred) {
    return std::sqrt(mse(y_true, y_pred));
}

Float nrmse(const Matrix& y_true, const Matrix& y_pred, 
            const std::string& normalization) {
    check_arrays(y_true, y_pred);
    
    Float rmse_val = rmse(y_true, y_pred);
    Float normalizer;
    
    if (normalization == "var") {
        // Normalize by variance
        Float mean_y = y_true.mean();
        Matrix centered = y_true.array() - mean_y;
        normalizer = std::sqrt(centered.array().square().mean());
    } else if (normalization == "std") {
        // Normalize by standard deviation  
        Float mean_y = y_true.mean();
        Matrix centered = y_true.array() - mean_y;
        normalizer = std::sqrt(centered.array().square().mean());
    } else if (normalization == "range") {
        // Normalize by range (max - min)
        Float min_y = y_true.minCoeff();
        Float max_y = y_true.maxCoeff();
        normalizer = max_y - min_y;
    } else if (normalization == "mean") {
        // Normalize by mean of absolute values
        normalizer = y_true.array().abs().mean();
    } else {
        throw std::invalid_argument("Unknown normalization method: " + normalization);
    }
    
    if (normalizer == 0.0) {
        throw std::invalid_argument("Normalization factor is zero");
    }
    
    return rmse_val / normalizer;
}

Float rsquare(const Matrix& y_true, const Matrix& y_pred) {
    check_arrays(y_true, y_pred);
    
    // R² = 1 - SS_res / SS_tot
    // SS_res = Σ(y_true - y_pred)²
    // SS_tot = Σ(y_true - mean(y_true))²
    
    Float mean_y = y_true.mean();
    Matrix y_mean_centered = y_true.array() - mean_y;
    
    Float ss_res = (y_true - y_pred).array().square().sum();
    Float ss_tot = y_mean_centered.array().square().sum();
    
    if (ss_tot == 0.0) {
        // Perfect fit case (constant true values)
        return (ss_res == 0.0) ? 1.0 : 0.0;
    }
    
    return 1.0 - (ss_res / ss_tot);
}

Float spectral_radius(const Matrix& W, int max_iter) {
    (void)max_iter; // Suppress unused parameter warning
    
    if (W.rows() != W.cols()) {
        throw std::invalid_argument("Matrix must be square for spectral radius computation");
    }
    
    if (W.rows() == 0) {
        return 0.0;
    }
    
    // Use Eigen to compute eigenvalues
    Eigen::EigenSolver<Matrix> solver(W);
    
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigenvalue computation failed");
    }
    
    // Get complex eigenvalues and compute their magnitudes
    auto eigenvalues = solver.eigenvalues();
    Float max_abs_eigenvalue = 0.0;
    
    for (int i = 0; i < eigenvalues.size(); ++i) {
        Float magnitude = std::abs(eigenvalues(i));
        max_abs_eigenvalue = std::max(max_abs_eigenvalue, magnitude);
    }
    
    return max_abs_eigenvalue;
}

Float effective_spectral_radius(const Matrix& states, int n_samples) {
    if (states.rows() < 2) {
        throw std::invalid_argument("Need at least 2 time steps to compute effective spectral radius");
    }
    
    int time_steps = states.rows();
    
    // Limit samples to available time steps
    int actual_samples = std::min(n_samples, time_steps - 1);
    
    // Compute norms of consecutive states
    Float sum_ratio = 0.0;
    int valid_samples = 0;
    
    for (int t = 1; t < time_steps && valid_samples < actual_samples; ++t) {
        Vector state_prev = states.row(t-1);
        Vector state_curr = states.row(t);
        
        Float norm_prev = state_prev.norm();
        Float norm_curr = state_curr.norm();
        
        if (norm_prev > 1e-12) { // Avoid division by very small numbers
            sum_ratio += norm_curr / norm_prev;
            valid_samples++;
        }
    }
    
    if (valid_samples == 0) {
        return 0.0;
    }
    
    return sum_ratio / valid_samples;
}

Float memory_capacity(const Matrix& reservoir_states, 
                     const Matrix& input_history,
                     int max_delay) {
    if (reservoir_states.rows() != input_history.rows()) {
        throw std::invalid_argument("Reservoir states and input history must have same number of time steps");
    }
    
    if (input_history.cols() != 1) {
        throw std::invalid_argument("Memory capacity currently supports only 1D input signals");
    }
    
    int time_steps = reservoir_states.rows();
    int n_nodes = reservoir_states.cols();
    
    if (time_steps <= max_delay) {
        throw std::invalid_argument("Time series too short for requested max_delay");
    }
    
    Float total_capacity = 0.0;
    
    // For each delay k, compute the linear readout that best reconstructs input[t-k]
    for (int k = 1; k <= max_delay; ++k) {
        if (time_steps <= k) break;
        
        // Prepare training data: states[k:] -> input[:-k] 
        int train_samples = time_steps - k;
        Matrix X = reservoir_states.bottomRows(train_samples);
        Vector y = input_history.topRows(train_samples).col(0);
        
        // Add bias term to X
        Matrix X_bias(train_samples, n_nodes + 1);
        X_bias.leftCols(n_nodes) = X;
        X_bias.rightCols(1) = Matrix::Ones(train_samples, 1);
        
        // Solve linear regression: w = (X^T X + reg I)^{-1} X^T y
        Float regularization = 1e-3;  // Increased regularization to prevent overfitting
        Matrix XTX = X_bias.transpose() * X_bias;
        Vector XTy = X_bias.transpose() * y;
        
        // Normalize regularization by data scale
        Float data_scale = XTX.diagonal().mean();
        Matrix I = Matrix::Identity(XTX.rows(), XTX.cols());
        Matrix regularized_XTX = XTX + (regularization * data_scale) * I;
        
        Vector weights = regularized_XTX.llt().solve(XTy);
        
        // Make predictions
        Vector y_pred = X_bias * weights;
        
        // Compute R² for this delay
        Float r_squared = rsquare(y.transpose(), y_pred.transpose());
        
        // Memory capacity is sum of R² values
        total_capacity += std::max(0.0, r_squared); // Ensure non-negative
    }
    
    return total_capacity;
}

} // namespace observables
} // namespace reservoircpp