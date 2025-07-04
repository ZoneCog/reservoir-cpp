/**
 * @file reservoir.cpp
 * @brief Implementation of reservoir classes
 */

#include "reservoircpp/reservoir.hpp"
#include <stdexcept>

namespace reservoircpp {

Reservoir::Reservoir(const std::string& name, int units, Float lr,
                     const std::string& activation, Float connectivity,
                     Float spectral_radius, Float input_scaling,
                     Float bias_scaling)
    : Node(name), units_(units), lr_(lr), connectivity_(connectivity),
      spectral_radius_(spectral_radius), input_scaling_(input_scaling),
      bias_scaling_(bias_scaling), activation_name_(activation),
      use_internal_activation_(true) {
    
    if (units <= 0) {
        throw std::invalid_argument("Number of units must be positive");
    }
    if (lr <= 0.0 || lr > 1.0) {
        throw std::invalid_argument("Leak rate must be between 0 and 1");
    }
    if (connectivity <= 0.0 || connectivity > 1.0) {
        throw std::invalid_argument("Connectivity must be between 0 and 1");
    }
    if (spectral_radius <= 0.0) {
        throw std::invalid_argument("Spectral radius must be positive");
    }
    
    // Get activation function
    try {
        activation_fn_ = activations::get_function(activation_name_);
    } catch (const std::exception& e) {
        throw std::invalid_argument("Invalid activation function: " + activation_name_);
    }
    
    // Set output dimension
    set_output_dim({units_});
    
    // Initialize state using the public interface
    Vector initial_state = Vector::Zero(units_);
    this->reset(&initial_state);
    internal_state_ = Matrix::Zero(units_, 1);
}

void Reservoir::initialize(const Matrix* x, const Matrix* y) {
    if (x != nullptr) {
        // Set input dimension from data
        set_input_dim({static_cast<int>(x->cols())});
    }
    
    if (input_dim().empty()) {
        throw std::runtime_error("Input dimension must be set before initialization");
    }
    
    // Initialize weight matrices
    initialize_weights();
    
    // Mark as initialized using public interface
    // The Node class should provide a protected method to set this
    // For now, we'll rely on the base class implementation
}

void Reservoir::reset(const Vector* state) {
    if (state != nullptr) {
        if (state->size() != units_) {
            throw std::invalid_argument("State size mismatch");
        }
        set_state(*state);
        internal_state_ = *state;
    } else {
        Vector zero_state = Vector::Zero(units_);
        set_state(zero_state);
        internal_state_ = Matrix::Zero(units_, 1);
    }
}

Matrix Reservoir::forward(const Matrix& x) {
    if (!is_initialized()) {
        throw std::runtime_error("Reservoir must be initialized before forward pass");
    }
    
    if (x.cols() != input_dim()[0]) {
        throw std::invalid_argument("Input dimension mismatch");
    }
    
    Matrix output(units_, x.rows());
    
    for (int t = 0; t < x.rows(); ++t) {
        Matrix u = x.row(t).transpose();
        
        if (use_internal_activation_) {
            output.col(t) = forward_internal(u);
        } else {
            output.col(t) = forward_external(u);
        }
    }
    
    return output.transpose();
}

std::shared_ptr<Node> Reservoir::copy(const std::string& name) const {
    auto copy = std::make_shared<Reservoir>(name, units_, lr_, activation_name_,
                                           connectivity_, spectral_radius_,
                                           input_scaling_, bias_scaling_);
    
    // Copy state if initialized
    if (is_initialized()) {
        copy->W_ = W_;
        copy->W_in_ = W_in_;
        copy->bias_ = bias_;
        Vector current_state = get_state();
        copy->set_state(current_state);
        copy->internal_state_ = internal_state_;
        copy->set_input_dim(input_dim());
        copy->set_output_dim(output_dim());
    }
    
    return copy;
}

void Reservoir::initialize_weights() {
    int input_size = input_dim()[0];
    
    // Generate recurrent weights
    W_ = matrix_generators::generate_internal_weights(units_, connectivity_, 
                                                     spectral_radius_, "uniform");
    
    // Generate input weights
    W_in_ = matrix_generators::generate_input_weights(units_, input_size, 
                                                     input_scaling_, 1.0, "uniform");
    
    // Generate bias
    if (bias_scaling_ > 0.0) {
        bias_ = matrix_generators::uniform(units_, 1, -bias_scaling_, bias_scaling_);
    } else {
        bias_ = Matrix::Zero(units_, 1);
    }
}

Matrix Reservoir::reservoir_kernel(const Matrix& u, const Vector& r) {
    // Compute: W * r + W_in * u + bias
    // Convert vector to matrix for computation
    Matrix r_mat = r;
    Matrix pre_activation = W_ * r_mat + W_in_ * u + bias_;
    return pre_activation;
}

Matrix Reservoir::forward_internal(const Matrix& x) {
    // Forward pass with internal activation:
    // r[t+1] = (1 - lr) * r[t] + lr * f(W_in * u[t] + W * r[t] + bias)
    
    Vector r = get_state();
    Matrix pre_activation = reservoir_kernel(x, r);
    Matrix activated = activation_fn_(pre_activation);
    
    // Update state with leak rate
    Vector new_state = (1.0 - lr_) * r + lr_ * activated.col(0);
    set_state(new_state);
    
    return new_state;
}

Matrix Reservoir::forward_external(const Matrix& x) {
    // Forward pass with external activation:
    // s[t+1] = (1 - lr) * s[t] + lr * (W_in * u[t] + W * r[t] + bias)
    // r[t+1] = f(s[t+1])
    
    Matrix s = internal_state_;
    Vector r = get_state();
    Matrix pre_activation = reservoir_kernel(x, r);
    
    // Update internal state
    internal_state_ = (1.0 - lr_) * s + lr_ * pre_activation;
    
    // Apply activation to get output state
    Matrix activated = activation_fn_(internal_state_);
    Vector new_state = activated.col(0);
    set_state(new_state);
    
    return new_state;
}

// ESN implementation
ESN::ESN(const std::string& name, int units, Float lr, Float connectivity,
         Float spectral_radius, Float input_scaling, Float bias_scaling)
    : Reservoir(name, units, lr, "tanh", connectivity, spectral_radius,
                input_scaling, bias_scaling) {
    // ESN uses tanh activation by default
}

std::shared_ptr<Node> ESN::copy(const std::string& name) const {
    auto copy = std::make_shared<ESN>(name, units_, lr_, connectivity_,
                                     spectral_radius_, input_scaling_, bias_scaling_);
    
    // Copy state if initialized
    if (is_initialized()) {
        copy->W_ = W_;
        copy->W_in_ = W_in_;
        copy->bias_ = bias_;
        Vector current_state = get_state();
        copy->set_state(current_state);
        copy->internal_state_ = internal_state_;
        copy->set_input_dim(input_dim());
        copy->set_output_dim(output_dim());
    }
    
    return copy;
}

} // namespace reservoircpp