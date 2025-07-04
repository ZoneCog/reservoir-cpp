/**
 * @file activations.hpp
 * @brief Activation functions for ReservoirCpp
 * 
 * This file contains the C++ port of the Python activationsfunc.py,
 * providing efficient implementations of common activation functions.
 */

#ifndef RESERVOIRCPP_ACTIVATIONS_HPP
#define RESERVOIRCPP_ACTIVATIONS_HPP

#include <Eigen/Dense>
#include <functional>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>

#include "reservoircpp/types.hpp"

namespace reservoircpp {
namespace activations {

/**
 * @brief Identity activation function
 * 
 * f(x) = x
 * 
 * @param x Input matrix
 * @return Output matrix (same as input)
 */
inline Matrix identity(const Matrix& x) {
    return x;
}

/**
 * @brief Sigmoid activation function
 * 
 * f(x) = 1 / (1 + exp(-x))
 * 
 * @param x Input matrix
 * @return Output matrix with sigmoid applied element-wise
 */
inline Matrix sigmoid(const Matrix& x) {
    return x.unaryExpr([](Float val) -> Float {
        if (val < 0) {
            Float exp_val = std::exp(val);
            return exp_val / (exp_val + 1.0);
        }
        return 1.0 / (1.0 + std::exp(-val));
    });
}

/**
 * @brief Hyperbolic tangent activation function
 * 
 * f(x) = tanh(x)
 * 
 * @param x Input matrix
 * @return Output matrix with tanh applied element-wise
 */
inline Matrix tanh(const Matrix& x) {
    return x.unaryExpr([](Float val) -> Float {
        return std::tanh(val);
    });
}

/**
 * @brief ReLU activation function
 * 
 * f(x) = x if x > 0, else 0
 * 
 * @param x Input matrix
 * @return Output matrix with ReLU applied element-wise
 */
inline Matrix relu(const Matrix& x) {
    return x.unaryExpr([](Float val) -> Float {
        return std::max(0.0, val);
    });
}

/**
 * @brief Softplus activation function
 * 
 * f(x) = ln(1 + exp(x))
 * 
 * @param x Input matrix
 * @return Output matrix with softplus applied element-wise
 */
inline Matrix softplus(const Matrix& x) {
    return x.unaryExpr([](Float val) -> Float {
        return std::log(1.0 + std::exp(val));
    });
}

/**
 * @brief Softmax activation function
 * 
 * Applies softmax across each row of the input matrix
 * 
 * @param x Input matrix
 * @param beta Temperature parameter (default 1.0)
 * @return Output matrix with softmax applied row-wise
 */
inline Matrix softmax(const Matrix& x, Float beta = 1.0) {
    Matrix result = x;
    
    for (int i = 0; i < x.rows(); ++i) {
        Vector row = beta * x.row(i).transpose();
        
        // Numerical stability: subtract max value
        Float max_val = row.maxCoeff();
        row = row.array() - max_val;
        
        // Compute exponentials
        Vector exp_row = row.unaryExpr([](Float val) -> Float {
            return std::exp(val);
        });
        
        // Normalize
        Float sum = exp_row.sum();
        result.row(i) = exp_row.transpose() / sum;
    }
    
    return result;
}

/**
 * @brief Function registry for activation functions
 * 
 * Equivalent to the get_function() in Python
 */
class ActivationRegistry {
public:
    using ActivationFn = std::function<Matrix(const Matrix&)>;
    using ActivationFnBeta = std::function<Matrix(const Matrix&, Float)>;
    
    static ActivationRegistry& instance() {
        static ActivationRegistry registry;
        return registry;
    }
    
    /**
     * @brief Get activation function by name
     * 
     * @param name Function name (identity, sigmoid, tanh, relu, softplus, softmax)
     * @return Activation function
     * @throws std::invalid_argument if function name is not found
     */
    ActivationFn get_function(const std::string& name) const {
        auto it = functions_.find(name);
        if (it == functions_.end()) {
            throw std::invalid_argument("Unknown activation function: " + name);
        }
        return it->second;
    }
    
    /**
     * @brief Get softmax function with beta parameter
     * 
     * @param beta Temperature parameter
     * @return Softmax function with specified beta
     */
    ActivationFn get_softmax(Float beta = 1.0) const {
        return [beta](const Matrix& x) -> Matrix {
            return softmax(x, beta);
        };
    }
    
    /**
     * @brief Get list of available function names
     * 
     * @return Vector of function names
     */
    std::vector<std::string> available_functions() const {
        std::vector<std::string> names;
        for (const auto& pair : functions_) {
            names.push_back(pair.first);
        }
        return names;
    }
    
private:
    ActivationRegistry() {
        // Register all activation functions
        functions_["identity"] = identity;
        functions_["id"] = identity;
        functions_["sigmoid"] = sigmoid;
        functions_["sig"] = sigmoid;
        functions_["tanh"] = tanh;
        functions_["relu"] = relu;
        functions_["re"] = relu;
        functions_["softplus"] = softplus;
        functions_["sp"] = softplus;
        functions_["softmax"] = [](const Matrix& x) -> Matrix { return softmax(x); };
        functions_["smax"] = [](const Matrix& x) -> Matrix { return softmax(x); };
    }
    
    std::unordered_map<std::string, ActivationFn> functions_;
};

/**
 * @brief Convenience function to get activation function by name
 * 
 * @param name Function name
 * @return Activation function
 */
inline ActivationRegistry::ActivationFn get_function(const std::string& name) {
    return ActivationRegistry::instance().get_function(name);
}

} // namespace activations
} // namespace reservoircpp

#endif // RESERVOIRCPP_ACTIVATIONS_HPP