/**
 * @file reservoir.hpp
 * @brief Base reservoir class for ReservoirCpp
 * 
 * This file contains the C++ port of the Python reservoir base,
 * implementing the core reservoir functionality.
 */

#ifndef RESERVOIRCPP_RESERVOIR_HPP
#define RESERVOIRCPP_RESERVOIR_HPP

#include <memory>
#include <string>
#include <functional>

#include "reservoircpp/types.hpp"
#include "reservoircpp/node.hpp"
#include "reservoircpp/activations.hpp"
#include "reservoircpp/matrix_generators.hpp"

namespace reservoircpp {

/**
 * @brief Base reservoir class
 * 
 * Implements a pool of leaky-integrator neurons with random recurrent connections.
 * The reservoir follows the equation:
 * 
 * x[t+1] = (1 - lr) * x[t] + lr * f(W_in * u[t] + W * x[t] + bias)
 * 
 * where:
 * - x[t] is the reservoir state
 * - u[t] is the input
 * - lr is the leak rate
 * - f is the activation function
 * - W_in is the input weight matrix
 * - W is the recurrent weight matrix
 */
class Reservoir : public Node {
public:
    /**
     * @brief Construct a new Reservoir
     * 
     * @param name Node name
     * @param units Number of reservoir units
     * @param lr Leak rate (default: 1.0)
     * @param activation Activation function name (default: "tanh")
     * @param connectivity Connectivity ratio (default: 0.1)
     * @param spectral_radius Spectral radius (default: 0.9)
     * @param input_scaling Input scaling factor (default: 1.0)
     * @param bias_scaling Bias scaling factor (default: 0.0)
     */
    Reservoir(const std::string& name, int units, Float lr = 1.0,
              const std::string& activation = "tanh", Float connectivity = 0.1,
              Float spectral_radius = 0.9, Float input_scaling = 1.0,
              Float bias_scaling = 0.0);

    /**
     * @brief Initialize the reservoir
     * 
     * @param x Input data for initialization (optional)
     * @param y Target data for initialization (optional)
     */
    void initialize(const Matrix* x = nullptr, const Matrix* y = nullptr) override;

    /**
     * @brief Reset the reservoir state
     * 
     * @param state New state (optional, defaults to zeros)
     */
    void reset(const Vector* state = nullptr) override;

    /**
     * @brief Forward pass through the reservoir
     * 
     * @param x Input data
     * @return Reservoir output
     */
    Matrix forward(const Matrix& x) override;

    /**
     * @brief Copy the reservoir
     * 
     * @param name New name
     * @return Copied reservoir
     */
    std::shared_ptr<Node> copy(const std::string& name) const override;

    // Getters
    int units() const { return units_; }
    Float leak_rate() const { return lr_; }
    Float connectivity() const { return connectivity_; }
    Float spectral_radius() const { return spectral_radius_; }
    Float input_scaling() const { return input_scaling_; }
    Float bias_scaling() const { return bias_scaling_; }
    const std::string& activation_name() const { return activation_name_; }
    bool is_reservoir_initialized() const { return reservoir_initialized_; }
    
    // Weight matrices
    const Matrix& W() const { return W_; }
    const Matrix& W_in() const { return W_in_; }
    const Matrix& bias() const { return bias_; }
    
    // State
    const Matrix& internal_state() const { return internal_state_; }

    // Setters
    void set_leak_rate(Float lr) { lr_ = lr; }
    void set_connectivity(Float connectivity) { connectivity_ = connectivity; }
    void set_spectral_radius(Float sr) { spectral_radius_ = sr; }
    void set_input_scaling(Float scaling) { input_scaling_ = scaling; }
    void set_bias_scaling(Float scaling) { bias_scaling_ = scaling; }

protected:
    /**
     * @brief Initialize weight matrices
     */
    virtual void initialize_weights();
    
    /**
     * @brief Override the do_initialize method from Node
     */
    void do_initialize(const Matrix* x, const Matrix* y) override;

    /**
     * @brief Reservoir kernel computation
     * 
     * @param u Input vector
     * @param r Current state vector
     * @return Pre-activation values
     */
    virtual Matrix reservoir_kernel(const Matrix& u, const Vector& r);

    /**
     * @brief Forward pass with internal activation
     * 
     * @param x Input
     * @return Output
     */
    Matrix forward_internal(const Matrix& x);

    /**
     * @brief Forward pass with external activation
     * 
     * @param x Input
     * @return Output
     */
    Matrix forward_external(const Matrix& x);

    // Parameters
    int units_;
    Float lr_;
    Float connectivity_;
    Float spectral_radius_;
    Float input_scaling_;
    Float bias_scaling_;
    std::string activation_name_;
    
    // Weight matrices
    Matrix W_;       // Recurrent weights
    Matrix W_in_;    // Input weights
    Matrix bias_;    // Bias vector
    
    // State
    Matrix internal_state_;  // Internal state for external activation
    
    // Activation function
    activations::ActivationRegistry::ActivationFn activation_fn_;
    
    // Flags
    bool use_internal_activation_;
    bool reservoir_initialized_;  // Track our own initialization
};

/**
 * @brief Echo State Network (ESN) class
 * 
 * A specific type of reservoir with standard ESN properties.
 */
class ESN : public Reservoir {
public:
    /**
     * @brief Construct a new ESN
     * 
     * @param name Node name
     * @param units Number of reservoir units
     * @param lr Leak rate (default: 1.0)
     * @param connectivity Connectivity ratio (default: 0.1)
     * @param spectral_radius Spectral radius (default: 0.9)
     * @param input_scaling Input scaling factor (default: 1.0)
     * @param bias_scaling Bias scaling factor (default: 0.0)
     */
    ESN(const std::string& name, int units, Float lr = 1.0,
        Float connectivity = 0.1, Float spectral_radius = 0.9,
        Float input_scaling = 1.0, Float bias_scaling = 0.0);

    /**
     * @brief Copy the ESN
     * 
     * @param name New name
     * @return Copied ESN
     */
    std::shared_ptr<Node> copy(const std::string& name) const override;
};

} // namespace reservoircpp

#endif // RESERVOIRCPP_RESERVOIR_HPP