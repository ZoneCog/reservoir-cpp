/**
 * @file experimental.cpp
 * @brief Implementation of experimental features and nodes
 */

#include "reservoircpp/experimental.hpp"
#include "reservoircpp/utils.hpp"
#include "reservoircpp/activations.hpp"
#include <cmath>
#include <algorithm>
#include <random>

namespace reservoircpp {
namespace experimental {

// LIF Implementation
LIF::LIF(const std::string& name, size_t units, float tau_m, float tau_s, 
         float threshold, float reset, float dt)
    : Node(name, {{"tau_m", tau_m}, {"tau_s", tau_s}, {"threshold", threshold}, 
                  {"reset", reset}, {"dt", dt}}), 
      tau_m_(tau_m), tau_s_(tau_s), threshold_(threshold), 
      reset_(reset), dt_(dt) {
    
    set_output_dim({static_cast<int>(units)});
}

void LIF::initialize() {
    if (output_dim().empty()) {
        throw std::runtime_error("LIF: output dimension not set");
    }
    
    size_t units = output_dim()[0];
    
    // Initialize state matrices
    membrane_potential_ = Matrix::Zero(1, units);
    synaptic_current_ = Matrix::Zero(1, units);
    spike_output_ = Matrix::Zero(1, units);
}

void LIF::reset_state() {
    if (!is_initialized()) {
        return;
    }
    
    membrane_potential_.setZero();
    synaptic_current_.setZero();
    spike_output_.setZero();
}

Matrix LIF::forward(const Matrix& input) {
    if (!is_initialized()) {
        initialize();
    }
    
    if (input.cols() != membrane_potential_.cols()) {
        throw std::runtime_error("LIF: input size mismatch");
    }
    
    // LIF dynamics equations
    // dV/dt = (1/tau_m) * (-V + I_syn + I_ext)
    // dI_syn/dt = -I_syn/tau_s + input
    
    float alpha_m = dt_ / tau_m_;
    float alpha_s = dt_ / tau_s_;
    
    // Update synaptic current
    synaptic_current_ = (1.0f - alpha_s) * synaptic_current_ + alpha_s * input;
    
    // Update membrane potential
    membrane_potential_ = (1.0f - alpha_m) * membrane_potential_ + 
                         alpha_m * synaptic_current_;
    
    // Check for spikes and reset
    spike_output_.setZero();
    for (int i = 0; i < membrane_potential_.cols(); ++i) {
        if (membrane_potential_(0, i) >= threshold_) {
            spike_output_(0, i) = 1.0f;
            membrane_potential_(0, i) = reset_;
        }
    }
    
    return spike_output_;
}

// Add Implementation
Add::Add(const std::string& name) : Node(name) {
    // Add node doesn't have fixed output dimension - it depends on inputs
}

void Add::initialize() {
    // Nothing specific to initialize for Add node
}

Matrix Add::forward(const Matrix& input) {
    if (!has_second_input_) {
        throw std::runtime_error("Add: second input not set");
    }
    return forward(input, second_input_);
}

Matrix Add::forward(const Matrix& input1, const Matrix& input2) {
    if (input1.rows() != input2.rows() || input1.cols() != input2.cols()) {
        throw std::runtime_error("Add: input dimensions must match");
    }
    
    return input1 + input2;
}

void Add::set_second_input(const Matrix& input2) {
    second_input_ = input2;
    has_second_input_ = true;
}

// BatchFORCE Implementation
BatchFORCE::BatchFORCE(const std::string& name, size_t output_dim, float alpha)
    : Node(name, {{"alpha", alpha}}), alpha_(alpha) {
    
    set_output_dim({static_cast<int>(output_dim)});
}
}

void BatchFORCE::initialize() {
    if (output_dim().empty()) {
        throw std::runtime_error("BatchFORCE: output dimension not set");
    }
    
    // Initialize weights randomly (will be resized when first input arrives)
    weights_ = Matrix::Zero(output_dim()[0], 1);
    P_ = Matrix::Identity(1, 1) / alpha_;
}

void BatchFORCE::reset_state() {
    if (!is_initialized()) {
        return;
    }
    
    // Reset inverse correlation matrix
    if (weights_.cols() > 0) {
        P_ = Matrix::Identity(weights_.cols(), weights_.cols()) / alpha_;
    }
}

Matrix BatchFORCE::forward(const Matrix& input) {
    if (!has_target_) {
        throw std::runtime_error("BatchFORCE: target not set for training");
    }
    return forward(input, target_);
}

Matrix BatchFORCE::forward(const Matrix& input, const Matrix& target) {
    if (!is_initialized()) {
        initialize();
    }
    
    size_t output_dim = this->output_dim()[0];
    
    // Initialize weights if this is the first input
    if (weights_.cols() != input.cols()) {
        weights_ = Matrix::Random(output_dim, input.cols()) * 0.1f;
        P_ = Matrix::Identity(input.cols(), input.cols()) / alpha_;
    }
    
    if (target.rows() != output_dim || target.cols() != input.cols()) {
        throw std::runtime_error("BatchFORCE: target dimension mismatch");
    }
    
    // Forward pass
    Matrix output = weights_ * input.transpose();
    
    // FORCE learning update
    Matrix error = target.transpose() - output;
    Matrix k = P_ * input.transpose();
    Matrix rPr = input * k;
    
    // Update P (inverse correlation matrix)
    Matrix k_outer = k * input;
    float denominator = 1.0f + rPr(0, 0);
    P_ = P_ - (k_outer) / denominator;
    
    // Update weights
    weights_ = weights_ + error * k.transpose() / denominator;
    
    return output.transpose();
}

void BatchFORCE::set_target(const Matrix& target) {
    target_ = target;
    has_target_ = true;
}

// RandomChoice Implementation
RandomChoice::RandomChoice(const std::string& name, size_t n_features, unsigned int seed)
    : Node(name, {{"n_features", static_cast<float>(n_features)}, 
                  {"seed", static_cast<float>(seed)}}), 
      n_features_(n_features), seed_(seed) {
}

void RandomChoice::initialize() {
    // Indices will be initialized when first input arrives
}

void RandomChoice::set_n_features(size_t n_features) {
    n_features_ = n_features;
    initialized_indices_ = false;  // Force reinitialization
}

void RandomChoice::set_seed(unsigned int seed) {
    seed_ = seed;
    initialized_indices_ = false;  // Force reinitialization
}

Matrix RandomChoice::forward(const Matrix& input) {
    if (!is_initialized()) {
        initialize();
    }
    
    // Initialize indices if needed
    if (!initialized_indices_ || indices_.empty()) {
        size_t input_features = input.cols();
        
        if (n_features_ > input_features) {
            throw std::runtime_error("RandomChoice: n_features cannot exceed input size");
        }
        
        // Generate random indices
        indices_.clear();
        for (size_t i = 0; i < input_features; ++i) {
            indices_.push_back(i);
        }
        
        std::mt19937 rng(seed_);
        std::shuffle(indices_.begin(), indices_.end(), rng);
        indices_.resize(n_features_);
        
        initialized_indices_ = true;
        set_output_dim({static_cast<int>(n_features_)});
    }
    
    // Select features
    Matrix output(input.rows(), n_features_);
    for (size_t i = 0; i < n_features_; ++i) {
        output.col(i) = input.col(indices_[i]);
    }
    
    return output;
}

} // namespace experimental
} // namespace reservoircpp