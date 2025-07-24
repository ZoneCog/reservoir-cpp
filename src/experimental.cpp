/**
 * @file experimental_simple.cpp
 * @brief Simplified implementation of experimental features
 */

#include "reservoircpp/experimental.hpp"
#include "reservoircpp/utils.hpp"
#include <cmath>
#include <algorithm>
#include <random>

namespace reservoircpp {
namespace experimental {

// Simplified LIF Implementation
LIF::LIF(const std::string& name, size_t units, float tau_m, float tau_s, 
         float threshold, float reset, float dt)
    : Node(name), tau_m_(tau_m), tau_s_(tau_s), threshold_(threshold), 
      reset_(reset), dt_(dt) {
    
    set_output_dim({static_cast<int>(units)});
}

void LIF::initialize() {
    if (output_dim().empty()) {
        throw std::runtime_error("LIF: output dimension not set");
    }
    
    size_t units = output_dim()[0];
    membrane_potential_ = Matrix::Zero(1, units);
    synaptic_current_ = Matrix::Zero(1, units);
    spike_output_ = Matrix::Zero(1, units);
}

void LIF::reset_state() {
    if (membrane_potential_.size() > 0) {
        membrane_potential_.setZero();
        synaptic_current_.setZero();
        spike_output_.setZero();
    }
}

Matrix LIF::forward(const Matrix& input) {
    if (membrane_potential_.size() == 0) {
        initialize();
    }
    
    if (input.cols() != membrane_potential_.cols()) {
        throw std::runtime_error("LIF: input size mismatch");
    }
    
    // Simplified LIF dynamics
    float alpha_m = dt_ / tau_m_;
    float alpha_s = dt_ / tau_s_;
    
    synaptic_current_ = (1.0f - alpha_s) * synaptic_current_ + alpha_s * input;
    membrane_potential_ = (1.0f - alpha_m) * membrane_potential_ + alpha_m * synaptic_current_;
    
    spike_output_.setZero();
    for (int i = 0; i < membrane_potential_.cols(); ++i) {
        if (membrane_potential_(0, i) >= threshold_) {
            spike_output_(0, i) = 1.0f;
            membrane_potential_(0, i) = reset_;
        }
    }
    
    return spike_output_;
}

// Simplified Add Implementation
Add::Add(const std::string& name) : Node(name) {}

void Add::initialize() {}

Matrix Add::forward(const Matrix& input) {
    if (!has_second_input_) {
        throw std::runtime_error("Add: second input not set");
    }
    return input + second_input_;
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

// Simplified BatchFORCE Implementation  
BatchFORCE::BatchFORCE(const std::string& name, size_t output_dim, float alpha)
    : Node(name), alpha_(alpha) {
    set_output_dim({static_cast<int>(output_dim)});
}

void BatchFORCE::initialize() {
    if (output_dim().empty()) {
        throw std::runtime_error("BatchFORCE: output dimension not set");
    }
    weights_ = Matrix::Zero(output_dim()[0], 1);
    P_ = Matrix::Identity(1, 1) / alpha_;
}

void BatchFORCE::reset_state() {
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
    if (weights_.size() == 0) {
        initialize();
    }
    
    size_t output_dim_val = output_dim()[0];
    
    if (weights_.cols() != input.cols()) {
        weights_ = Matrix::Random(output_dim_val, input.cols()) * 0.1f;
        P_ = Matrix::Identity(input.cols(), input.cols()) / alpha_;
    }
    
    if (target.rows() != output_dim_val || target.cols() != input.cols()) {
        throw std::runtime_error("BatchFORCE: target dimension mismatch");
    }
    
    Matrix output = weights_ * input.transpose();
    Matrix error = target.transpose() - output;
    Matrix k = P_ * input.transpose();
    
    // Simplified FORCE update
    weights_ = weights_ + error * k.transpose() / (1.0f + (input * k)(0, 0));
    
    return output.transpose();
}

void BatchFORCE::set_target(const Matrix& target) {
    target_ = target;
    has_target_ = true;
}

// Simplified RandomChoice Implementation
RandomChoice::RandomChoice(const std::string& name, size_t n_features, unsigned int seed)
    : Node(name), n_features_(n_features), seed_(seed) {}

void RandomChoice::initialize() {}

void RandomChoice::set_n_features(size_t n_features) {
    n_features_ = n_features;
    initialized_indices_ = false;
}

void RandomChoice::set_seed(unsigned int seed) {
    seed_ = seed;
    initialized_indices_ = false;
}

Matrix RandomChoice::forward(const Matrix& input) {
    if (!initialized_indices_ || indices_.empty()) {
        size_t input_features = input.cols();
        
        if (n_features_ > input_features) {
            throw std::runtime_error("RandomChoice: n_features cannot exceed input size");
        }
        
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
    
    Matrix output(input.rows(), n_features_);
    for (size_t i = 0; i < n_features_; ++i) {
        output.col(i) = input.col(indices_[i]);
    }
    
    return output;
}

} // namespace experimental
} // namespace reservoircpp