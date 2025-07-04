/**
 * @file readout.cpp
 * @brief Implementation of readout classes
 */

#include "reservoircpp/readout.hpp"
#include <Eigen/Dense>
#include <stdexcept>

namespace reservoircpp {

// Base Readout class
Readout::Readout(const std::string& name, int output_dim, bool input_bias)
    : Node(name), input_bias_(input_bias), is_fitted_(false), readout_initialized_(false) {
    if (output_dim <= 0) {
        throw std::invalid_argument("Output dimension must be positive");
    }
    
    set_output_dim({output_dim});
    // The Node base class is already trainable by default
}

void Readout::initialize(const Matrix* x, const Matrix* y) {
    if (readout_initialized_) {
        return; // Already initialized
    }
    
    if (x != nullptr) {
        int input_size = x->cols();
        if (input_bias_) {
            input_size += 1;  // Add bias term
        }
        set_input_dim({input_size});
    }
    
    if (input_dim().empty()) {
        throw std::runtime_error("Input dimension must be set before initialization");
    }
    
    initialize_weights();
    readout_initialized_ = true;
}

void Readout::reset(const Vector* state) {
    is_fitted_ = false;
    if (readout_initialized_) {
        initialize_weights();
    }
    if (state != nullptr) {
        set_state(*state);
    } else {
        Vector zero_state = Vector::Zero(output_dim()[0]);
        set_state(zero_state);
    }
}

Matrix Readout::forward(const Matrix& x) {
    if (!is_fitted_) {
        throw std::runtime_error("Readout must be fitted before forward pass");
    }
    
    return predict(x);
}

Matrix Readout::predict(const Matrix& x) {
    if (!is_fitted_) {
        throw std::runtime_error("Readout must be fitted before prediction");
    }
    
    Matrix processed_x = prepare_inputs(x);
    return (W_out_ * processed_x.transpose()).transpose();
}

void Readout::partial_fit(const Matrix& x, const Matrix& y) {
    // Default implementation: just call fit
    fit(x, y);
}

Matrix Readout::prepare_inputs(const Matrix& x) {
    if (input_bias_) {
        // Add bias column
        Matrix processed_x(x.rows(), x.cols() + 1);
        processed_x.leftCols(x.cols()) = x;
        processed_x.rightCols(1) = Matrix::Ones(x.rows(), 1);
        return processed_x;
    } else {
        return x;
    }
}

void Readout::initialize_weights() {
    int input_size = input_dim()[0];
    int output_size = output_dim()[0];
    
    W_out_ = matrix_generators::normal(output_size, input_size, 0.0, 0.1);
    
    if (input_bias_) {
        bias_ = Matrix::Zero(output_size, 1);
    }
}

// Ridge Readout
RidgeReadout::RidgeReadout(const std::string& name, int output_dim, Float ridge,
                          bool input_bias)
    : Readout(name, output_dim, input_bias), ridge_(ridge) {
    if (ridge <= 0.0) {
        throw std::invalid_argument("Ridge parameter must be positive");
    }
}

void RidgeReadout::fit(const Matrix& x, const Matrix& y) {
    if (x.rows() != y.rows()) {
        throw std::invalid_argument("Input and target must have same number of samples");
    }
    
    if (y.cols() != output_dim()[0]) {
        throw std::invalid_argument("Target dimension mismatch");
    }
    
    Matrix processed_x = prepare_inputs(x);
    
    // Solve ridge regression: W = (X^T X + ridge * I)^{-1} X^T Y
    Matrix XTX = processed_x.transpose() * processed_x;
    Matrix XTY = processed_x.transpose() * y;
    
    // Add ridge regularization
    Matrix I = Matrix::Identity(XTX.rows(), XTX.cols());
    Matrix regularized_XTX = XTX + ridge_ * I;
    
    // Solve the system
    W_out_ = regularized_XTX.llt().solve(XTY).transpose();
    
    is_fitted_ = true;
}

std::shared_ptr<Node> RidgeReadout::copy(const std::string& name) const {
    auto copy = std::make_shared<RidgeReadout>(name, output_dim()[0], ridge_, input_bias_);
    
    if (readout_initialized_) {
        copy->W_out_ = W_out_;
        copy->bias_ = bias_;
        copy->is_fitted_ = is_fitted_;
        copy->readout_initialized_ = true;
        copy->set_input_dim(input_dim());
        copy->set_output_dim(output_dim());
    }
    
    return copy;
}

// FORCE Readout
ForceReadout::ForceReadout(const std::string& name, int output_dim, Float learning_rate,
                          Float regularization, bool input_bias)
    : Readout(name, output_dim, input_bias), learning_rate_(learning_rate),
      regularization_(regularization) {
    if (learning_rate <= 0.0 || learning_rate > 1.0) {
        throw std::invalid_argument("Learning rate must be between 0 and 1");
    }
    if (regularization <= 0.0) {
        throw std::invalid_argument("Regularization parameter must be positive");
    }
}

void ForceReadout::initialize(const Matrix* x, const Matrix* y) {
    Readout::initialize(x, y);
    
    if (readout_initialized_) {
        // Initialize inverse correlation matrix
        int input_size = input_dim()[0];
        P_ = Matrix::Identity(input_size, input_size) / regularization_;
    }
}

void ForceReadout::fit(const Matrix& x, const Matrix& y) {
    if (x.rows() != y.rows()) {
        throw std::invalid_argument("Input and target must have same number of samples");
    }
    
    if (y.cols() != output_dim()[0]) {
        throw std::invalid_argument("Target dimension mismatch");
    }
    
    // Initialize if not done
    if (!readout_initialized_) {
        initialize(&x, &y);
    }
    
    // Batch FORCE learning
    for (int t = 0; t < x.rows(); ++t) {
        Matrix x_t = x.row(t).transpose();
        Matrix y_t = y.row(t).transpose();
        
        partial_fit(x_t.transpose(), y_t.transpose());
    }
    
    is_fitted_ = true;
}

void ForceReadout::partial_fit(const Matrix& x, const Matrix& y) {
    if (x.rows() != 1 || y.rows() != 1) {
        throw std::invalid_argument("Partial fit expects single sample");
    }
    
    Matrix processed_x = prepare_inputs(x);
    Matrix r = processed_x.transpose();  // Column vector
    Matrix z = y.transpose();            // Column vector
    
    // FORCE update equations
    Matrix k = P_ * r;
    Matrix rPr = r.transpose() * k;
    Float c = 1.0 / (1.0 + rPr(0, 0));
    
    // Update P matrix
    P_ = P_ - c * k * k.transpose();
    
    // Update weights
    Matrix e = W_out_ * r - z;
    W_out_ = W_out_ - c * e * k.transpose();
    
    is_fitted_ = true;
}

std::shared_ptr<Node> ForceReadout::copy(const std::string& name) const {
    auto copy = std::make_shared<ForceReadout>(name, output_dim()[0], learning_rate_,
                                              regularization_, input_bias_);
    
    if (readout_initialized_) {
        copy->W_out_ = W_out_;
        copy->bias_ = bias_;
        copy->P_ = P_;
        copy->is_fitted_ = is_fitted_;
        copy->readout_initialized_ = true;
        copy->set_input_dim(input_dim());
        copy->set_output_dim(output_dim());
    }
    
    return copy;
}

// LMS Readout
LMSReadout::LMSReadout(const std::string& name, int output_dim, Float learning_rate,
                      bool input_bias)
    : Readout(name, output_dim, input_bias), learning_rate_(learning_rate) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
}

void LMSReadout::fit(const Matrix& x, const Matrix& y) {
    if (x.rows() != y.rows()) {
        throw std::invalid_argument("Input and target must have same number of samples");
    }
    
    if (y.cols() != output_dim()[0]) {
        throw std::invalid_argument("Target dimension mismatch");
    }
    
    // Initialize if not done
    if (!readout_initialized_) {
        initialize(&x, &y);
    }
    
    // Batch LMS learning
    for (int t = 0; t < x.rows(); ++t) {
        Matrix x_t = x.row(t).transpose();
        Matrix y_t = y.row(t).transpose();
        
        partial_fit(x_t.transpose(), y_t.transpose());
    }
    
    is_fitted_ = true;
}

void LMSReadout::partial_fit(const Matrix& x, const Matrix& y) {
    if (x.rows() != 1 || y.rows() != 1) {
        throw std::invalid_argument("Partial fit expects single sample");
    }
    
    Matrix processed_x = prepare_inputs(x);
    Matrix r = processed_x.transpose();  // Column vector
    Matrix z = y.transpose();            // Column vector
    
    // LMS update: W = W + lr * e * r^T
    Matrix predicted = W_out_ * r;
    Matrix error = z - predicted;
    
    W_out_ = W_out_ + learning_rate_ * error * r.transpose();
    
    is_fitted_ = true;
}

std::shared_ptr<Node> LMSReadout::copy(const std::string& name) const {
    auto copy = std::make_shared<LMSReadout>(name, output_dim()[0], learning_rate_,
                                            input_bias_);
    
    if (readout_initialized_) {
        copy->W_out_ = W_out_;
        copy->bias_ = bias_;
        copy->is_fitted_ = is_fitted_;
        copy->readout_initialized_ = true;
        copy->set_input_dim(input_dim());
        copy->set_output_dim(output_dim());
    }
    
    return copy;
}

} // namespace reservoircpp