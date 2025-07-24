/**
 * @file reservoir.cpp
 * @brief Implementation of reservoir classes
 */

#include "reservoircpp/reservoir.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <functional>

namespace reservoircpp {

Reservoir::Reservoir(const std::string& name, int units, Float lr,
                     const std::string& activation, Float connectivity,
                     Float spectral_radius, Float input_scaling,
                     Float bias_scaling)
    : Node(name), units_(units), lr_(lr), connectivity_(connectivity),
      spectral_radius_(spectral_radius), input_scaling_(input_scaling),
      bias_scaling_(bias_scaling), activation_name_(activation),
      use_internal_activation_(true), reservoir_initialized_(false) {
    
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
    if (reservoir_initialized_) {
        return; // Already initialized
    }
    
    if (x != nullptr) {
        // For reservoirs, input dimension is the number of features (columns)
        set_input_dim({static_cast<int>(x->cols())});
    }
    
    if (input_dim().empty()) {
        throw std::runtime_error("Input dimension must be set before initialization");
    }
    
    // Set output dimension to number of units
    set_output_dim({units_});
    
    // Initialize weight matrices
    initialize_weights();
    
    // Initialize state
    Vector zero_state = Vector::Zero(units_);
    set_state(zero_state);
    internal_state_ = Matrix::Zero(units_, 1);
    
    // Mark as initialized
    reservoir_initialized_ = true;
}

void Reservoir::do_initialize(const Matrix* x, const Matrix* y) {
    // This is called by Node::initialize, but we've already done our initialization
    // Just do nothing here since we handled everything in our initialize() method
    (void)x; // Suppress unused parameter warning
    (void)y;
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
    if (!reservoir_initialized_) {
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
    if (reservoir_initialized_) {
        copy->W_ = W_;
        copy->W_in_ = W_in_;
        copy->bias_ = bias_;
        Vector current_state = get_state();
        copy->set_state(current_state);
        copy->internal_state_ = internal_state_;
        copy->set_input_dim(input_dim());
        copy->set_output_dim(output_dim());
        copy->reservoir_initialized_ = true;
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
    if (reservoir_initialized_) {
        copy->W_ = W_;
        copy->W_in_ = W_in_;
        copy->bias_ = bias_;
        Vector current_state = get_state();
        copy->set_state(current_state);
        copy->internal_state_ = internal_state_;
        copy->set_input_dim(input_dim());
        copy->set_output_dim(output_dim());
        copy->reservoir_initialized_ = true;
    }
    
    return copy;
}

// IntrinsicPlasticity implementation
IntrinsicPlasticity::IntrinsicPlasticity(const std::string& name, int units,
                                       Float lr, Float mu, Float sigma,
                                       Float learning_rate, int epochs,
                                       const std::string& activation,
                                       Float connectivity, Float spectral_radius,
                                       Float input_scaling, Float bias_scaling)
    : Reservoir(name, units, lr, activation, connectivity, spectral_radius,
                input_scaling, bias_scaling),
      mu_(mu), sigma_(sigma), learning_rate_(learning_rate), epochs_(epochs),
      fitted_(false) {
    
    // Validate activation function for IP
    if (activation != "tanh" && activation != "sigmoid") {
        throw std::invalid_argument("IntrinsicPlasticity activation must be 'tanh' or 'sigmoid'");
    }
    
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    
    if (epochs <= 0) {
        throw std::invalid_argument("Number of epochs must be positive");
    }
    
    // Initialize IP parameters
    a_ = Matrix::Ones(units_, 1);
    b_ = Matrix::Zero(units_, 1);
    
    // Use external activation for IP
    use_internal_activation_ = false;
}

void IntrinsicPlasticity::initialize(const Matrix* x, const Matrix* y) {
    // Call parent initialization
    Reservoir::initialize(x, y);
    
    // Reset IP parameters
    a_ = Matrix::Ones(units_, 1);
    b_ = Matrix::Zero(units_, 1);
    fitted_ = false;
}

Matrix IntrinsicPlasticity::forward(const Matrix& x) {
    if (!reservoir_initialized_) {
        throw std::runtime_error("IntrinsicPlasticity must be initialized before forward pass");
    }
    
    if (x.cols() != input_dim()[0]) {
        throw std::invalid_argument("Input dimension mismatch");
    }
    
    Matrix output(units_, x.rows());
    
    for (int t = 0; t < x.rows(); ++t) {
        Matrix u = x.row(t).transpose();
        
        // Forward pass with external activation and IP
        Matrix s = internal_state_;
        Vector r = get_state();
        Matrix pre_activation = reservoir_kernel(u, r);
        
        // Update internal state
        internal_state_ = (1.0 - lr_) * s + lr_ * pre_activation;
        
        // Apply IP activation: f(a * s + b)
        Matrix activated = ip_activation(internal_state_);
        Vector new_state = activated.col(0);
        set_state(new_state);
        
        output.col(t) = new_state;
    }
    
    return output.transpose();
}

void IntrinsicPlasticity::fit(const std::vector<Matrix>& x, int warmup) {
    if (!reservoir_initialized_) {
        throw std::runtime_error("IntrinsicPlasticity must be initialized before fitting");
    }
    
    for (int epoch = 0; epoch < epochs_; ++epoch) {
        for (const auto& seq : x) {
            // Reset state for each sequence
            reset();
            
            // Warmup phase
            for (int t = 0; t < std::min(warmup, static_cast<int>(seq.rows())); ++t) {
                Matrix u = seq.row(t).transpose();
                
                // Forward pass without learning
                Matrix s = internal_state_;
                Vector r = get_state();
                Matrix pre_activation = reservoir_kernel(u, r);
                internal_state_ = (1.0 - lr_) * s + lr_ * pre_activation;
                Matrix activated = ip_activation(internal_state_);
                set_state(activated.col(0));
            }
            
            // Learning phase
            for (int t = warmup; t < seq.rows(); ++t) {
                Matrix u = seq.row(t).transpose();
                
                // Store pre-activation state
                Vector r = get_state();
                Matrix pre_state = internal_state_;
                
                // Forward pass
                Matrix kernel_output = reservoir_kernel(u, r);
                internal_state_ = (1.0 - lr_) * internal_state_ + lr_ * kernel_output;
                Matrix post_state = ip_activation(internal_state_);
                set_state(post_state.col(0));
                
                // Update IP parameters
                update_ip_parameters(pre_state, post_state);
            }
        }
    }
    
    fitted_ = true;
}

void IntrinsicPlasticity::partial_fit(const Matrix& x_batch, int warmup) {
    if (!reservoir_initialized_) {
        throw std::runtime_error("IntrinsicPlasticity must be initialized before partial fitting");
    }
    
    // Reset state
    reset();
    
    // Warmup phase
    for (int t = 0; t < std::min(warmup, static_cast<int>(x_batch.rows())); ++t) {
        Matrix u = x_batch.row(t).transpose();
        
        // Forward pass without learning
        Matrix s = internal_state_;
        Vector r = get_state();
        Matrix pre_activation = reservoir_kernel(u, r);
        internal_state_ = (1.0 - lr_) * s + lr_ * pre_activation;
        Matrix activated = ip_activation(internal_state_);
        set_state(activated.col(0));
    }
    
    // Learning phase
    for (int t = warmup; t < x_batch.rows(); ++t) {
        Matrix u = x_batch.row(t).transpose();
        
        // Store pre-activation state
        Vector r = get_state();
        Matrix pre_state = internal_state_;
        
        // Forward pass
        Matrix kernel_output = reservoir_kernel(u, r);
        internal_state_ = (1.0 - lr_) * internal_state_ + lr_ * kernel_output;
        Matrix post_state = ip_activation(internal_state_);
        set_state(post_state.col(0));
        
        // Update IP parameters
        update_ip_parameters(pre_state, post_state);
    }
    
    fitted_ = true;
}

std::shared_ptr<Node> IntrinsicPlasticity::copy(const std::string& name) const {
    auto copy = std::make_shared<IntrinsicPlasticity>(name, units_, lr_, mu_, sigma_,
                                                     learning_rate_, epochs_,
                                                     activation_name_, connectivity_,
                                                     spectral_radius_, input_scaling_,
                                                     bias_scaling_);
    
    // Copy state if initialized
    if (reservoir_initialized_) {
        copy->W_ = W_;
        copy->W_in_ = W_in_;
        copy->bias_ = bias_;
        copy->a_ = a_;
        copy->b_ = b_;
        Vector current_state = get_state();
        copy->set_state(current_state);
        copy->internal_state_ = internal_state_;
        copy->set_input_dim(input_dim());
        copy->set_output_dim(output_dim());
        copy->reservoir_initialized_ = true;
        copy->fitted_ = fitted_;
    }
    
    return copy;
}

void IntrinsicPlasticity::update_ip_parameters(const Matrix& pre_state, const Matrix& post_state) {
    std::pair<Matrix, Matrix> gradients;
    
    if (activation_name_ == "tanh") {
        gradients = gaussian_gradients(pre_state.transpose(), post_state.transpose(),
                                     a_, mu_, sigma_, learning_rate_);
    } else { // sigmoid
        gradients = exp_gradients(pre_state.transpose(), post_state.transpose(),
                                a_, mu_, learning_rate_);
    }
    
    // Apply gradients
    a_ += gradients.first.transpose();
    b_ += gradients.second.transpose();
}

std::pair<Matrix, Matrix> IntrinsicPlasticity::gaussian_gradients(const Matrix& x, const Matrix& y,
                                                                const Matrix& a, Float mu,
                                                                Float sigma, Float eta) {
    Float sig2 = sigma * sigma;
    
    // All operations in matrix form
    Matrix ones = Matrix::Ones(y.rows(), y.cols());
    Matrix y_sq = y.cwiseProduct(y);  // y^2
    
    Matrix delta_b = -eta * (-(mu / sig2) * ones + 
                            (y / sig2).cwiseProduct(2 * sig2 * ones + ones - y_sq + mu * y));
    
    Matrix a_inv = a.cwiseInverse();  // 1/a element-wise
    Matrix delta_a = eta * a_inv + delta_b.cwiseProduct(x);
    
    return {delta_a, delta_b};
}

std::pair<Matrix, Matrix> IntrinsicPlasticity::exp_gradients(const Matrix& x, const Matrix& y,
                                                           const Matrix& a, Float mu, Float eta) {
    Matrix ones = Matrix::Ones(y.rows(), y.cols());
    Matrix y_sq = y.cwiseProduct(y);  // y^2
    
    Matrix delta_b = eta * (ones - (2 + (1 / mu)) * y + y_sq / mu);
    
    Matrix a_inv = a.cwiseInverse();  // 1/a element-wise
    Matrix delta_a = eta * a_inv + delta_b.cwiseProduct(x);
    
    return {delta_a, delta_b};
}

Matrix IntrinsicPlasticity::ip_activation(const Matrix& state) {
    // Apply IP transformation: f(a * state + b)
    Matrix transformed = a_.cwiseProduct(state) + b_;
    return activation_fn_(transformed);
}

// NVAR implementation
NVAR::NVAR(const std::string& name, int delay, int order, int strides)
    : Node(name), delay_(delay), order_(order), strides_(strides),
      linear_dim_(0), nonlinear_dim_(0), nvar_initialized_(false) {
    
    if (delay <= 0) {
        throw std::invalid_argument("Delay must be positive");
    }
    if (order <= 0) {
        throw std::invalid_argument("Order must be positive");
    }
    if (strides <= 0) {
        throw std::invalid_argument("Strides must be positive");
    }
}

void NVAR::initialize(const Matrix* x, const Matrix* y) {
    if (nvar_initialized_) {
        return; // Already initialized
    }
    
    if (x != nullptr) {
        // For NVAR, input dimension is the number of features (columns)
        int input_dim = static_cast<int>(x->cols());
        set_input_dim({input_dim});
        
        // Compute dimensions
        linear_dim_ = delay_ * input_dim;
        nonlinear_dim_ = combinations_with_replacement(linear_dim_, order_);
        int output_dim = linear_dim_ + nonlinear_dim_;
        
        set_output_dim({output_dim});
        
        // Initialize storage for delayed inputs
        store_ = Matrix::Zero(delay_ * strides_, input_dim);
        
        // Generate monomial indices
        monomial_indices_ = generate_monomial_indices(linear_dim_, order_);
        
        nvar_initialized_ = true;
    } else {
        throw std::runtime_error("Input data required for NVAR initialization");
    }
}

void NVAR::do_initialize(const Matrix* x, const Matrix* y) {
    // This is called by Node::initialize, but we've already done our initialization
    // Just do nothing here since we handled everything in our initialize() method
    (void)x; // Suppress unused parameter warning
    (void)y;
}

void NVAR::reset(const Vector* state) {
    if (nvar_initialized_) {
        // Reset the input storage
        store_.setZero();
    }
    
    // NVAR doesn't have an internal state like reservoirs, so we just reset storage
    (void)state; // NVAR doesn't use external state
}

Matrix NVAR::forward(const Matrix& x) {
    if (!nvar_initialized_) {
        throw std::runtime_error("NVAR must be initialized before forward pass");
    }
    
    if (x.cols() != input_dim()[0]) {
        throw std::invalid_argument("Input dimension mismatch");
    }
    
    Matrix output(x.rows(), output_dim()[0]);
    
    for (int t = 0; t < x.rows(); ++t) {
        Vector u = x.row(t).transpose();
        
        // Update storage: roll and add new input
        // Roll storage down by 1 (move each row down)
        for (int i = store_.rows() - 1; i > 0; --i) {
            store_.row(i) = store_.row(i - 1);
        }
        store_.row(0) = u.transpose();
        
        // Extract linear features with strides
        Vector linear_feats(linear_dim_);
        int feat_idx = 0;
        for (int d = 0; d < delay_; ++d) {
            int store_idx = d * strides_;
            if (store_idx < store_.rows()) {
                for (int j = 0; j < input_dim()[0]; ++j) {
                    linear_feats(feat_idx++) = store_(store_idx, j);
                }
            } else {
                // Pad with zeros if we don't have enough history
                for (int j = 0; j < input_dim()[0]; ++j) {
                    linear_feats(feat_idx++) = 0.0;
                }
            }
        }
        
        // Compute nonlinear features
        Vector nonlinear_feats = compute_monomials(linear_feats);
        
        // Combine linear and nonlinear features
        Vector combined_output(linear_dim_ + nonlinear_dim_);
        combined_output.head(linear_dim_) = linear_feats;
        combined_output.tail(nonlinear_dim_) = nonlinear_feats;
        
        output.row(t) = combined_output.transpose();
    }
    
    return output;
}

std::shared_ptr<Node> NVAR::copy(const std::string& name) const {
    auto copy = std::make_shared<NVAR>(name, delay_, order_, strides_);
    
    // Copy state if initialized
    if (nvar_initialized_) {
        copy->linear_dim_ = linear_dim_;
        copy->nonlinear_dim_ = nonlinear_dim_;
        copy->store_ = store_;
        copy->monomial_indices_ = monomial_indices_;
        copy->set_input_dim(input_dim());
        copy->set_output_dim(output_dim());
        copy->nvar_initialized_ = true;
    }
    
    return copy;
}

int NVAR::combinations_with_replacement(int n, int k) {
    // Compute C(n+k-1, k) = C(n+k-1, n-1)
    if (k == 0) return 1;
    if (n == 1) return 1;
    
    // Use formula: C(n,k) = n! / (k! * (n-k)!)
    // For combinations with replacement: C(n+k-1, k)
    long long result = 1;
    int numerator = n + k - 1;
    int denominator = std::min(k, numerator - k);
    
    for (int i = 0; i < denominator; ++i) {
        result = result * (numerator - i) / (i + 1);
    }
    
    return static_cast<int>(result);
}

std::vector<std::vector<int>> NVAR::generate_monomial_indices(int linear_dim, int order) {
    std::vector<std::vector<int>> indices;
    
    // Generate all combinations with replacement of linear_dim elements taken order at a time
    std::function<void(std::vector<int>&, int, int)> generate_combinations = 
        [&](std::vector<int>& current, int start, int remaining) {
            if (remaining == 0) {
                indices.push_back(current);
                return;
            }
            
            for (int i = start; i < linear_dim; ++i) {
                current.push_back(i);
                generate_combinations(current, i, remaining - 1);
                current.pop_back();
            }
        };
    
    std::vector<int> current;
    generate_combinations(current, 0, order);
    
    return indices;
}

Vector NVAR::compute_monomials(const Vector& linear_feats) {
    Vector monomials(nonlinear_dim_);
    
    for (size_t i = 0; i < monomial_indices_.size(); ++i) {
        Float product = 1.0;
        for (int idx : monomial_indices_[i]) {
            product *= linear_feats(idx);
        }
        monomials(i) = product;
    }
    
    return monomials;
}

} // namespace reservoircpp