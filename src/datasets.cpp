/**
 * @file datasets.cpp
 * @brief Implementation of dataset generators for Reservoir Computing
 */

#include "reservoircpp/datasets.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace reservoircpp {
namespace datasets {

namespace {
    // Helper function for Runge-Kutta 4th order integration
    template<typename F>
    Float rk4_step(F func, Float x, Float h) {
        Float k1 = h * func(x);
        Float k2 = h * func(x + 0.5 * k1);
        Float k3 = h * func(x + 0.5 * k2);
        Float k4 = h * func(x + k3);
        return x + (k1 + 2*k2 + 2*k3 + k4) / 6.0;
    }
    
    // Mackey-Glass differential equation
    Float mg_equation(Float xt, Float xtau, Float a, Float b, Float n) {
        return -b * xt + a * xtau / (1.0 + std::pow(xtau, n));
    }
}

Matrix mackey_glass(int n_timesteps, int tau, Float a, Float b,
                   Float n, Float h, Float x0, int washout) {
    if (n_timesteps <= 0) {
        throw std::invalid_argument("Number of timesteps must be positive");
    }
    if (tau <= 0) {
        throw std::invalid_argument("Tau must be positive");
    }
    
    int total_steps = n_timesteps + washout;
    std::vector<Float> x(total_steps + tau, x0);
    
    // Generate the time series using RK4 integration
    for (int t = tau; t < total_steps + tau; ++t) {
        auto mg_func = [&](Float xt) {
            return mg_equation(xt, x[t - tau], a, b, n);
        };
        x[t] = rk4_step(mg_func, x[t-1], h);
    }
    
    // Extract the desired portion (after washout)
    Matrix result(n_timesteps, 1);
    for (int i = 0; i < n_timesteps; ++i) {
        result(i, 0) = x[washout + tau + i];
    }
    
    return result;
}

Matrix lorenz(int n_timesteps, Float dt, Float sigma, 
              Float rho, Float beta,
              Float x0, Float y0, Float z0, int washout) {
    if (n_timesteps <= 0) {
        throw std::invalid_argument("Number of timesteps must be positive");
    }
    
    Matrix result(n_timesteps, 3);
    
    Float x = x0, y = y0, z = z0;
    
    // Washout period
    for (int t = 0; t < washout; ++t) {
        Float dx = sigma * (y - x);
        Float dy = x * (rho - z) - y;
        Float dz = x * y - beta * z;
        
        x += dt * dx;
        y += dt * dy;
        z += dt * dz;
    }
    
    // Generate the actual time series
    for (int t = 0; t < n_timesteps; ++t) {
        result(t, 0) = x;
        result(t, 1) = y;
        result(t, 2) = z;
        
        Float dx = sigma * (y - x);
        Float dy = x * (rho - z) - y;
        Float dz = x * y - beta * z;
        
        x += dt * dx;
        y += dt * dy;
        z += dt * dz;
    }
    
    return result;
}

Matrix henon_map(int n_timesteps, Float a, Float b,
                Float x0, Float y0, int washout) {
    if (n_timesteps <= 0) {
        throw std::invalid_argument("Number of timesteps must be positive");
    }
    
    Matrix result(n_timesteps, 2);
    
    Float x = x0, y = y0;
    
    // Washout period
    for (int t = 0; t < washout; ++t) {
        Float x_new = 1.0 - a * x * x + y;
        Float y_new = b * x;
        x = x_new;
        y = y_new;
    }
    
    // Generate the actual time series
    for (int t = 0; t < n_timesteps; ++t) {
        result(t, 0) = x;
        result(t, 1) = y;
        
        Float x_new = 1.0 - a * x * x + y;
        Float y_new = b * x;
        x = x_new;
        y = y_new;
    }
    
    return result;
}

Matrix logistic_map(int n_timesteps, Float r, Float x0, int washout) {
    if (n_timesteps <= 0) {
        throw std::invalid_argument("Number of timesteps must be positive");
    }
    if (x0 <= 0.0 || x0 >= 1.0) {
        throw std::invalid_argument("Initial condition must be in (0, 1)");
    }
    
    Matrix result(n_timesteps, 1);
    
    Float x = x0;
    
    // Washout period
    for (int t = 0; t < washout; ++t) {
        x = r * x * (1.0 - x);
        // Prevent escape from [0,1] due to numerical issues
        x = std::max(0.0, std::min(1.0, x));
    }
    
    // Generate the actual time series
    for (int t = 0; t < n_timesteps; ++t) {
        result(t, 0) = x;
        x = r * x * (1.0 - x);
        // Prevent escape from [0,1] due to numerical issues
        x = std::max(0.0, std::min(1.0, x));
    }
    
    return result;
}

std::tuple<Matrix, Matrix> narma(int n_timesteps, int order, 
                                Float alpha, Float beta, 
                                Float gamma, Float delta, 
                                int washout) {
    if (n_timesteps <= 0) {
        throw std::invalid_argument("Number of timesteps must be positive");
    }
    if (order <= 0) {
        throw std::invalid_argument("NARMA order must be positive");
    }
    
    int total_steps = n_timesteps + washout + order;
    
    // Generate random input in [0, 0.5]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Float> dis(0.0, 0.5);
    
    std::vector<Float> u(total_steps);
    std::vector<Float> y(total_steps, 0.0);
    
    for (int t = 0; t < total_steps; ++t) {
        u[t] = dis(gen);
    }
    
    // Initialize first values
    for (int t = 0; t < order; ++t) {
        y[t] = 0.0;
    }
    
    // Generate NARMA series
    for (int t = order; t < total_steps; ++t) {
        Float sum_y = 0.0;
        for (int i = 1; i <= order; ++i) {
            sum_y += y[t - i];
        }
        
        y[t] = alpha * y[t-1] + beta * y[t-1] * sum_y + 
               gamma * u[t-order] * u[t-1] + delta;
    }
    
    // Extract results after washout
    Matrix input(n_timesteps, 1);
    Matrix target(n_timesteps, 1);
    
    for (int t = 0; t < n_timesteps; ++t) {
        input(t, 0) = u[washout + order + t];
        target(t, 0) = y[washout + order + t];
    }
    
    return std::make_tuple(input, target);
}

std::tuple<Matrix, Matrix> to_forecasting(const Matrix& timeseries, int forecast) {
    if (timeseries.rows() <= forecast) {
        throw std::invalid_argument("Time series too short for forecasting");
    }
    
    int n_samples = timeseries.rows() - forecast;
    int n_features = timeseries.cols();
    
    Matrix X(n_samples, n_features);
    Matrix y(n_samples, n_features);
    
    for (int t = 0; t < n_samples; ++t) {
        X.row(t) = timeseries.row(t);
        y.row(t) = timeseries.row(t + forecast);
    }
    
    return std::make_tuple(X, y);
}

std::tuple<Matrix, Matrix, Matrix, Matrix> to_forecasting_with_split(
    const Matrix& timeseries, int forecast, int test_size) {
    if (timeseries.rows() <= forecast + test_size) {
        throw std::invalid_argument("Time series too short for forecasting with test split");
    }
    
    auto [X, y] = to_forecasting(timeseries, forecast);
    
    int train_size = X.rows() - test_size;
    
    Matrix X_train = X.topRows(train_size);
    Matrix X_test = X.bottomRows(test_size);
    Matrix y_train = y.topRows(train_size);
    Matrix y_test = y.bottomRows(test_size);
    
    return std::make_tuple(X_train, X_test, y_train, y_test);
}

Matrix one_hot_encode(const std::vector<int>& labels, int num_classes) {
    if (labels.empty()) {
        throw std::invalid_argument("Labels vector cannot be empty");
    }
    
    if (num_classes == 0) {
        // Infer number of classes from data
        num_classes = *std::max_element(labels.begin(), labels.end()) + 1;
    }
    
    Matrix result = Matrix::Zero(labels.size(), num_classes);
    
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] < 0 || labels[i] >= num_classes) {
            throw std::invalid_argument("Label out of range");
        }
        result(i, labels[i]) = 1.0;
    }
    
    return result;
}

Matrix mso(int n_timesteps, const std::vector<Float>& frequencies, bool normalize) {
    if (n_timesteps <= 0) {
        throw std::invalid_argument("Number of timesteps must be positive");
    }
    if (frequencies.empty()) {
        throw std::invalid_argument("Frequencies vector cannot be empty");
    }
    
    Matrix result(n_timesteps, 1);
    
    for (int t = 0; t < n_timesteps; ++t) {
        Float sum = 0.0;
        for (Float freq : frequencies) {
            sum += std::sin(freq * t);
        }
        result(t, 0) = sum;
    }
    
    if (normalize) {
        Float min_val = result.minCoeff();
        Float max_val = result.maxCoeff();
        
        if (max_val > min_val) {
            result = 2.0 * (result.array() - min_val) / (max_val - min_val) - 1.0;
        }
    }
    
    return result;
}

Matrix mso2(int n_timesteps, bool normalize) {
    return mso(n_timesteps, {0.2, 0.311}, normalize);
}

Matrix mso8(int n_timesteps, bool normalize) {
    return mso(n_timesteps, {0.2, 0.311, 0.42, 0.51, 0.63, 0.74, 0.85, 0.97}, normalize);
}

} // namespace datasets
} // namespace reservoircpp