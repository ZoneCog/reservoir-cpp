/**
 * @file readout.hpp
 * @brief Base readout classes for ReservoirCpp
 * 
 * This file contains the C++ port of the Python readout functionality,
 * implementing offline and online readout methods.
 */

#ifndef RESERVOIRCPP_READOUT_HPP
#define RESERVOIRCPP_READOUT_HPP

#include <memory>
#include <string>

#include "reservoircpp/types.hpp"
#include "reservoircpp/node.hpp"
#include "reservoircpp/matrix_generators.hpp"

namespace reservoircpp {

/**
 * @brief Base class for readout nodes
 * 
 * Readout nodes learn to map reservoir states to target outputs.
 */
class Readout : public Node {
public:
    /**
     * @brief Construct a new Readout
     * 
     * @param name Node name
     * @param output_dim Output dimension
     * @param input_bias Whether to add bias term (default: true)
     */
    Readout(const std::string& name, int output_dim, bool input_bias = true);

    /**
     * @brief Initialize the readout
     * 
     * @param x Input data for initialization
     * @param y Target data for initialization
     */
    void initialize(const Matrix* x = nullptr, const Matrix* y = nullptr) override;

    /**
     * @brief Reset the readout (clears learned weights)
     * 
     * @param state New state (optional)
     */
    void reset(const Vector* state = nullptr) override;

    /**
     * @brief Forward pass through the readout
     * 
     * @param x Input data
     * @return Readout output
     */
    Matrix forward(const Matrix& x) override;

    /**
     * @brief Fit the readout to training data
     * 
     * @param x Input data (reservoir states)
     * @param y Target data
     */
    virtual void fit(const Matrix& x, const Matrix& y) = 0;

    /**
     * @brief Predict using the fitted readout
     * 
     * @param x Input data
     * @return Predictions
     */
    virtual Matrix predict(const Matrix& x);

    /**
     * @brief Partial fit for online learning
     * 
     * @param x Input data
     * @param y Target data
     */
    virtual void partial_fit(const Matrix& x, const Matrix& y);

    // Getters
    bool input_bias() const { return input_bias_; }
    const Matrix& W_out() const { return W_out_; }
    bool is_fitted() const { return is_fitted_; }
    bool is_readout_initialized() const { return readout_initialized_; }

protected:
    /**
     * @brief Prepare inputs for learning (add bias if needed)
     * 
     * @param x Input data
     * @return Processed input data
     */
    Matrix prepare_inputs(const Matrix& x);

    /**
     * @brief Initialize output weights
     */
    virtual void initialize_weights();

    // Parameters
    bool input_bias_;
    bool is_fitted_;
    bool readout_initialized_;  // Track our own initialization
    
    // Weight matrix
    Matrix W_out_;  // Output weights
    Matrix bias_;   // Bias vector
};

/**
 * @brief Ridge regression readout
 * 
 * Implements ridge regression (L2 regularization) for readout learning.
 */
class RidgeReadout : public Readout {
public:
    /**
     * @brief Construct a new Ridge Readout
     * 
     * @param name Node name
     * @param output_dim Output dimension
     * @param ridge Ridge parameter (default: 1e-8)
     * @param input_bias Whether to add bias term (default: true)
     */
    RidgeReadout(const std::string& name, int output_dim, Float ridge = 1e-8,
                 bool input_bias = true);

    /**
     * @brief Fit the ridge readout
     * 
     * @param x Input data (reservoir states)
     * @param y Target data
     */
    void fit(const Matrix& x, const Matrix& y) override;

    /**
     * @brief Copy the ridge readout
     * 
     * @param name New name
     * @return Copied readout
     */
    std::shared_ptr<Node> copy(const std::string& name) const override;

    // Getters
    Float ridge() const { return ridge_; }
    
    // Setters
    void set_ridge(Float ridge) { ridge_ = ridge; }

private:
    Float ridge_;
};

/**
 * @brief FORCE learning readout
 * 
 * Implements FORCE (First-Order Reduced and Controlled Error) learning.
 */
class ForceReadout : public Readout {
public:
    /**
     * @brief Construct a new FORCE Readout
     * 
     * @param name Node name
     * @param output_dim Output dimension
     * @param learning_rate Learning rate (default: 1.0)
     * @param regularization Regularization parameter (default: 1.0)
     * @param input_bias Whether to add bias term (default: true)
     */
    ForceReadout(const std::string& name, int output_dim, Float learning_rate = 1.0,
                 Float regularization = 1.0, bool input_bias = true);

    /**
     * @brief Initialize the FORCE readout
     * 
     * @param x Input data for initialization
     * @param y Target data for initialization
     */
    void initialize(const Matrix* x = nullptr, const Matrix* y = nullptr) override;

    /**
     * @brief Fit the FORCE readout (batch mode)
     * 
     * @param x Input data (reservoir states)
     * @param y Target data
     */
    void fit(const Matrix& x, const Matrix& y) override;

    /**
     * @brief Partial fit for online FORCE learning
     * 
     * @param x Input data
     * @param y Target data
     */
    void partial_fit(const Matrix& x, const Matrix& y) override;

    /**
     * @brief Copy the FORCE readout
     * 
     * @param name New name
     * @return Copied readout
     */
    std::shared_ptr<Node> copy(const std::string& name) const override;

    // Getters
    Float learning_rate() const { return learning_rate_; }
    Float regularization() const { return regularization_; }
    
    // Setters
    void set_learning_rate(Float lr) { learning_rate_ = lr; }
    void set_regularization(Float reg) { regularization_ = reg; }

private:
    Float learning_rate_;
    Float regularization_;
    Matrix P_;  // Inverse correlation matrix
};

/**
 * @brief LMS (Least Mean Squares) readout
 * 
 * Implements LMS adaptive filter for online learning.
 */
class LMSReadout : public Readout {
public:
    /**
     * @brief Construct a new LMS Readout
     * 
     * @param name Node name
     * @param output_dim Output dimension
     * @param learning_rate Learning rate (default: 0.01)
     * @param input_bias Whether to add bias term (default: true)
     */
    LMSReadout(const std::string& name, int output_dim, Float learning_rate = 0.01,
               bool input_bias = true);

    /**
     * @brief Fit the LMS readout (batch mode)
     * 
     * @param x Input data (reservoir states)
     * @param y Target data
     */
    void fit(const Matrix& x, const Matrix& y) override;

    /**
     * @brief Partial fit for online LMS learning
     * 
     * @param x Input data
     * @param y Target data
     */
    void partial_fit(const Matrix& x, const Matrix& y) override;

    /**
     * @brief Copy the LMS readout
     * 
     * @param name New name
     * @return Copied readout
     */
    std::shared_ptr<Node> copy(const std::string& name) const override;

    // Getters
    Float learning_rate() const { return learning_rate_; }
    
    // Setters
    void set_learning_rate(Float lr) { learning_rate_ = lr; }

private:
    Float learning_rate_;
};

} // namespace reservoircpp

#endif // RESERVOIRCPP_READOUT_HPP