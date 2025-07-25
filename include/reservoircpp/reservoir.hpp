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

/**
 * @brief Intrinsic Plasticity Reservoir
 * 
 * A reservoir with intrinsic plasticity learning that adjusts neuron gains
 * and biases to achieve desired output distributions.
 * 
 * Based on the work by Triesch (2005) and Schrauwen et al. (2008).
 */
class IntrinsicPlasticity : public Reservoir {
public:
    /**
     * @brief Construct a new IntrinsicPlasticity reservoir
     * 
     * @param name Node name
     * @param units Number of reservoir units
     * @param lr Leak rate (default: 1.0)
     * @param mu Target mean of output distribution (default: 0.0)
     * @param sigma Target variance for Gaussian distribution (default: 1.0)
     * @param learning_rate IP learning rate (default: 5e-4)
     * @param epochs Number of training epochs (default: 1)
     * @param activation Activation function ("tanh" or "sigmoid", default: "tanh")
     * @param connectivity Connectivity ratio (default: 0.1)
     * @param spectral_radius Spectral radius (default: 0.9)
     * @param input_scaling Input scaling factor (default: 1.0)
     * @param bias_scaling Bias scaling factor (default: 0.0)
     */
    IntrinsicPlasticity(const std::string& name, int units, 
                       Float lr = 1.0, Float mu = 0.0, Float sigma = 1.0,
                       Float learning_rate = 5e-4, int epochs = 1,
                       const std::string& activation = "tanh",
                       Float connectivity = 0.1, Float spectral_radius = 0.9,
                       Float input_scaling = 1.0, Float bias_scaling = 0.0);

    /**
     * @brief Initialize the IP reservoir
     * 
     * @param x Input data for initialization (optional)
     * @param y Target data for initialization (optional)
     */
    void initialize(const Matrix* x = nullptr, const Matrix* y = nullptr) override;

    /**
     * @brief Forward pass through the IP reservoir
     * 
     * @param x Input data
     * @return Reservoir output
     */
    Matrix forward(const Matrix& x) override;

    /**
     * @brief Fit the intrinsic plasticity parameters
     * 
     * @param x Input sequences
     * @param warmup Number of warmup timesteps
     */
    void fit(const std::vector<Matrix>& x, int warmup = 0);

    /**
     * @brief Partial fit method for online learning
     * 
     * @param x_batch Input batch
     * @param warmup Number of warmup timesteps
     */
    void partial_fit(const Matrix& x_batch, int warmup = 0);

    /**
     * @brief Copy the IP reservoir
     * 
     * @param name New name
     * @return Copied IP reservoir
     */
    std::shared_ptr<Node> copy(const std::string& name) const override;

    // Getters for IP-specific parameters
    Float mu() const { return mu_; }
    Float sigma() const { return sigma_; }
    Float learning_rate() const { return learning_rate_; }
    int epochs() const { return epochs_; }
    const Matrix& a() const { return a_; }
    const Matrix& b() const { return b_; }

private:
    /**
     * @brief Perform one step of intrinsic plasticity learning
     * 
     * @param pre_state Pre-activation state
     * @param post_state Post-activation state
     */
    void update_ip_parameters(const Matrix& pre_state, const Matrix& post_state);

    /**
     * @brief Compute gradients for Gaussian distribution (tanh activation)
     * 
     * @param x Input
     * @param y Output 
     * @param a Current gain parameters
     * @param mu Target mean
     * @param sigma Target variance
     * @param eta Learning rate
     * @return Pair of (delta_a, delta_b)
     */
    std::pair<Matrix, Matrix> gaussian_gradients(const Matrix& x, const Matrix& y,
                                                const Matrix& a, Float mu, 
                                                Float sigma, Float eta);

    /**
     * @brief Compute gradients for exponential distribution (sigmoid activation)
     * 
     * @param x Input
     * @param y Output
     * @param a Current gain parameters  
     * @param mu Target mean
     * @param eta Learning rate
     * @return Pair of (delta_a, delta_b)
     */
    std::pair<Matrix, Matrix> exp_gradients(const Matrix& x, const Matrix& y,
                                           const Matrix& a, Float mu, Float eta);

    /**
     * @brief Apply IP activation with current a and b parameters
     * 
     * @param state Input state
     * @return Activated output
     */
    Matrix ip_activation(const Matrix& state);

    // IP-specific parameters
    Float mu_;              // Target mean
    Float sigma_;           // Target variance (for Gaussian)
    Float learning_rate_;   // IP learning rate
    int epochs_;            // Number of training epochs
    
    // IP parameters
    Matrix a_;              // Gain parameters (units x 1)
    Matrix b_;              // Bias parameters (units x 1)
    
    // Training state
    bool fitted_;           // Whether the model has been fitted
};

/**
 * @brief NVAR (Nonlinear Vector Autoregressive) Node
 * 
 * NVAR creates features by combining delayed inputs with nonlinear transformations
 * using monomials of specified order. Based on Gauthier et al. (2021).
 */
class NVAR : public Node {
public:
    /**
     * @brief Construct a new NVAR node
     * 
     * @param name Node name
     * @param delay Maximum delay of inputs (k)
     * @param order Order of nonlinear monomials (n) 
     * @param strides Strides between delayed inputs (default: 1)
     */
    NVAR(const std::string& name, int delay, int order, int strides = 1);

    /**
     * @brief Initialize the NVAR node
     * 
     * @param x Input data for initialization (optional)
     * @param y Target data for initialization (optional)
     */
    void initialize(const Matrix* x = nullptr, const Matrix* y = nullptr) override;

    /**
     * @brief Reset the NVAR node state
     * 
     * @param state New state (optional, defaults to zeros)
     */
    void reset(const Vector* state = nullptr) override;

    /**
     * @brief Forward pass through the NVAR node
     * 
     * @param x Input data
     * @return NVAR output with linear and nonlinear features
     */
    Matrix forward(const Matrix& x) override;

    /**
     * @brief Copy the NVAR node
     * 
     * @param name New name
     * @return Copied NVAR node
     */
    std::shared_ptr<Node> copy(const std::string& name) const override;

    // Getters
    int delay() const { return delay_; }
    int order() const { return order_; }
    int strides() const { return strides_; }
    int linear_dim() const { return linear_dim_; }
    int nonlinear_dim() const { return nonlinear_dim_; }
    const Matrix& store() const { return store_; }

private:
    /**
     * @brief Override the do_initialize method from Node
     */
    void do_initialize(const Matrix* x, const Matrix* y) override;

    /**
     * @brief Compute combinations with replacement
     * 
     * @param n Total number of items
     * @param k Number to choose
     * @return Number of combinations
     */
    int combinations_with_replacement(int n, int k);

    /**
     * @brief Generate monomial indices for nonlinear features
     * 
     * @param linear_dim Number of linear features
     * @param order Order of monomials
     * @return Matrix of indices for each monomial
     */
    std::vector<std::vector<int>> generate_monomial_indices(int linear_dim, int order);

    /**
     * @brief Compute monomial features from linear features
     * 
     * @param linear_feats Linear feature vector
     * @return Nonlinear monomial features
     */
    Vector compute_monomials(const Vector& linear_feats);

    // NVAR parameters
    int delay_;             // Maximum delay of inputs
    int order_;             // Order of nonlinear monomials
    int strides_;           // Strides between delayed inputs
    
    // Computed dimensions
    int linear_dim_;        // Dimension of linear features
    int nonlinear_dim_;     // Dimension of nonlinear features
    
    // Storage and precomputed indices
    Matrix store_;          // Storage for delayed inputs (delay*strides x input_dim)
    std::vector<std::vector<int>> monomial_indices_;  // Precomputed monomial indices
    
    // Flags
    bool nvar_initialized_; // Track our own initialization
};

} // namespace reservoircpp

#endif // RESERVOIRCPP_RESERVOIR_HPP