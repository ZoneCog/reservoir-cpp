/**
 * @file types.hpp
 * @brief Core type definitions for ReservoirCpp
 * 
 * This file contains the equivalent of the Python type.py file,
 * defining core types and interfaces using modern C++ features.
 */

#ifndef RESERVOIRCPP_TYPES_HPP
#define RESERVOIRCPP_TYPES_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <any>

namespace reservoircpp {

// Global precision type - equivalent to global_dtype in Python
using Float = double;

// Basic linear algebra types using Eigen
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using SparseMatrix = Eigen::SparseMatrix<Float>;

// Type aliases for different matrix storage formats
using DenseMatrix = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>;
using RowMajorMatrix = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Smart pointer types for memory management
template<typename T>
using unique_ptr = std::unique_ptr<T>;

template<typename T>
using shared_ptr = std::shared_ptr<T>;

// Shape type for dimensions
using Shape = std::vector<int>;

// Data types for input/output
using Data = Matrix;
using DataSequence = std::vector<Matrix>;

// Parameter storage type
using ParameterMap = std::unordered_map<std::string, std::any>;

// Activation function type
using ActivationFunction = std::function<Matrix(const Matrix&)>;

// Forward declarations for node-related types
class Node;
using NodePtr = shared_ptr<Node>;

/**
 * @brief Abstract base interface for all nodes in the computational graph
 * 
 * This is the C++ equivalent of the Python NodeType Protocol
 */
class NodeInterface {
public:
    virtual ~NodeInterface() = default;
    
    // Core interface methods
    virtual Matrix operator()(const Matrix& input) = 0;
    virtual void initialize(const Matrix* x = nullptr, const Matrix* y = nullptr) = 0;
    virtual void reset(const Vector* state = nullptr) = 0;
    
    // Parameter access
    virtual std::any get_param(const std::string& name) const = 0;
    virtual void set_param(const std::string& name, const std::any& value) = 0;
    
    // Properties
    virtual std::string name() const = 0;
    virtual void set_name(const std::string& name) = 0;
    virtual bool is_initialized() const = 0;
    virtual bool is_trainable() const = 0;
    virtual Shape input_dim() const = 0;
    virtual Shape output_dim() const = 0;
    
    // State management
    virtual Vector get_state() const = 0;
    virtual void set_state(const Vector& state) = 0;
};

/**
 * @brief Type trait to check if a type is a valid activation function
 */
template<typename T>
struct is_activation_function {
    static constexpr bool value = std::is_convertible_v<T, ActivationFunction>;
};

} // namespace reservoircpp

#endif // RESERVOIRCPP_TYPES_HPP