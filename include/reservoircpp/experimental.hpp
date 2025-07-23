/**
 * @file experimental.hpp
 * @brief Experimental features and nodes for ReservoirCpp
 * 
 * This module contains experimental features and nodes that are under
 * development and testing for future releases.
 * 
 * @warning All features in the experimental module may still be under heavy
 * development and subject to change.
 * 
 * @author ReservoirCpp Development Team
 * @date 2024
 * @license MIT License
 */

#ifndef RESERVOIRCPP_EXPERIMENTAL_HPP
#define RESERVOIRCPP_EXPERIMENTAL_HPP

#include "reservoircpp/types.hpp"
#include "reservoircpp/node.hpp"

namespace reservoircpp {
namespace experimental {

/**
 * @brief Leaky Integrate-and-Fire (LIF) neuron model
 * 
 * Implementation of a spiking neuron model based on the LIF dynamics.
 * Used for Liquid State Machine and other spiking neural network architectures.
 */
class LIF : public Node {
public:
    /**
     * @brief Constructor for LIF neuron
     * @param name Name of the node
     * @param units Number of neurons
     * @param tau_m Membrane time constant (default: 10.0)
     * @param tau_s Synaptic time constant (default: 2.0)
     * @param threshold Spike threshold (default: 1.0)
     * @param reset Reset potential (default: 0.0)
     * @param dt Time step (default: 1.0)
     */
    LIF(const std::string& name = "LIF",
        size_t units = 100,
        float tau_m = 10.0f,
        float tau_s = 2.0f,
        float threshold = 1.0f,
        float reset = 0.0f,
        float dt = 1.0f);

    /**
     * @brief Forward pass through LIF neurons
     * @param input Input current to neurons
     * @return Spike output (binary matrix)
     */
    Matrix forward(const Matrix& input) override;

    /**
     * @brief Reset neuron states
     */
    void reset_state();

    /**
     * @brief Initialize neuron parameters
     */
    void initialize();

    // Getters
    float get_tau_m() const { return tau_m_; }
    float get_tau_s() const { return tau_s_; }
    float get_threshold() const { return threshold_; }
    float get_reset() const { return reset_; }
    float get_dt() const { return dt_; }

    // Setters
    void set_tau_m(float tau_m) { tau_m_ = tau_m; }
    void set_tau_s(float tau_s) { tau_s_ = tau_s; }
    void set_threshold(float threshold) { threshold_ = threshold; }
    void set_reset(float reset) { reset_ = reset; }
    void set_dt(float dt) { dt_ = dt; }

private:
    float tau_m_;        ///< Membrane time constant
    float tau_s_;        ///< Synaptic time constant
    float threshold_;    ///< Spike threshold
    float reset_;        ///< Reset potential
    float dt_;           ///< Time step

    Matrix membrane_potential_;  ///< Current membrane potential
    Matrix synaptic_current_;   ///< Current synaptic current
    Matrix spike_output_;       ///< Spike output buffer
};

/**
 * @brief Add node - element-wise addition of two input vectors
 * 
 * Simple utility node that adds two input vectors element-wise.
 */
class Add : public Node {
public:
    /**
     * @brief Constructor for Add node
     * @param name Name of the node
     */
    explicit Add(const std::string& name = "Add");

    /**
     * @brief Forward pass - adds two inputs
     * @param input1 First input vector
     * @param input2 Second input vector (optional, can be set as parameter)
     * @return Element-wise sum of inputs
     */
    Matrix forward(const Matrix& input1, const Matrix& input2);
    Matrix forward(const Matrix& input) override;

    /**
     * @brief Set the second input as a parameter
     * @param input2 Second input to add
     */
    void set_second_input(const Matrix& input2);

    void initialize();

private:
    Matrix second_input_;  ///< Optional second input stored as parameter
    bool has_second_input_ = false;  ///< Flag indicating if second input is set
};

/**
 * @brief BatchFORCE - Fast implementation of FORCE learning algorithm
 * 
 * Efficient batch implementation of the FORCE learning algorithm
 * for online training of reservoir networks.
 */
class BatchFORCE : public Node {
public:
    /**
     * @brief Constructor for BatchFORCE
     * @param name Name of the node
     * @param output_dim Output dimensionality
     * @param alpha Learning rate (default: 1e-6)
     */
    BatchFORCE(const std::string& name = "BatchFORCE",
               size_t output_dim = 1,
               float alpha = 1e-6f);

    /**
     * @brief Forward pass and online learning
     * @param input Input from reservoir
     * @param target Target output for learning
     * @return Predicted output
     */
    Matrix forward(const Matrix& input, const Matrix& target);
    Matrix forward(const Matrix& input) override;

    /**
     * @brief Set target for next forward pass
     * @param target Target output
     */
    void set_target(const Matrix& target);

    void initialize();
    void reset_state();

    // Getters
    float get_alpha() const { return alpha_; }
    Matrix get_weights() const { return weights_; }

    // Setters
    void set_alpha(float alpha) { alpha_ = alpha; }

private:
    float alpha_;        ///< Learning rate
    Matrix weights_;     ///< Output weights
    Matrix P_;           ///< Inverse correlation matrix
    Matrix target_;      ///< Current target
    bool has_target_ = false;  ///< Flag for target availability
};

/**
 * @brief RandomChoice - Randomly select features from input vector
 * 
 * Utility node that randomly selects a subset of features from the input.
 */
class RandomChoice : public Node {
public:
    /**
     * @brief Constructor for RandomChoice
     * @param name Name of the node
     * @param n_features Number of features to select
     * @param seed Random seed
     */
    RandomChoice(const std::string& name = "RandomChoice",
                 size_t n_features = 10,
                 unsigned int seed = 42);

    Matrix forward(const Matrix& input) override;
    void initialize();

    // Getters
    size_t get_n_features() const { return n_features_; }
    unsigned int get_seed() const { return seed_; }

    // Setters
    void set_n_features(size_t n_features);
    void set_seed(unsigned int seed);

private:
    size_t n_features_;      ///< Number of features to select
    unsigned int seed_;      ///< Random seed
    std::vector<size_t> indices_;  ///< Selected feature indices
    bool initialized_indices_ = false;  ///< Flag for indices initialization
};

} // namespace experimental
} // namespace reservoircpp

#endif // RESERVOIRCPP_EXPERIMENTAL_HPP