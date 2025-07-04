/**
 * @file test_activations.cpp
 * @brief Tests for activation functions
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <reservoircpp/activations.hpp>
#include <cmath>

using namespace reservoircpp;
using namespace reservoircpp::activations;

TEST_CASE("Activation functions", "[activations]") {
    Matrix x(2, 3);
    x << 1.0, 0.0, -1.0,
         2.0, -0.5, 0.5;
    
    SECTION("Identity function") {
        Matrix result = identity(x);
        
        REQUIRE(result.rows() == x.rows());
        REQUIRE(result.cols() == x.cols());
        
        for (int i = 0; i < x.rows(); ++i) {
            for (int j = 0; j < x.cols(); ++j) {
                REQUIRE(result(i, j) == Catch::Approx(x(i, j)));
            }
        }
    }
    
    SECTION("Sigmoid function") {
        Matrix result = sigmoid(x);
        
        REQUIRE(result.rows() == x.rows());
        REQUIRE(result.cols() == x.cols());
        
        // Check that all values are in [0, 1]
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                REQUIRE(result(i, j) >= 0.0);
                REQUIRE(result(i, j) <= 1.0);
            }
        }
        
        // Check specific values
        REQUIRE(result(0, 0) == Catch::Approx(1.0 / (1.0 + std::exp(-1.0))));
        REQUIRE(result(0, 1) == Catch::Approx(0.5)); // sigmoid(0) = 0.5
        REQUIRE(result(0, 2) == Catch::Approx(1.0 / (1.0 + std::exp(1.0))));
    }
    
    SECTION("Tanh function") {
        Matrix result = tanh(x);
        
        REQUIRE(result.rows() == x.rows());
        REQUIRE(result.cols() == x.cols());
        
        // Check that all values are in [-1, 1]
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                REQUIRE(result(i, j) >= -1.0);
                REQUIRE(result(i, j) <= 1.0);
            }
        }
        
        // Check specific values
        REQUIRE(result(0, 0) == Catch::Approx(std::tanh(1.0)));
        REQUIRE(result(0, 1) == Catch::Approx(0.0)); // tanh(0) = 0
        REQUIRE(result(0, 2) == Catch::Approx(std::tanh(-1.0)));
    }
    
    SECTION("ReLU function") {
        Matrix result = relu(x);
        
        REQUIRE(result.rows() == x.rows());
        REQUIRE(result.cols() == x.cols());
        
        // Check specific values
        REQUIRE(result(0, 0) == Catch::Approx(1.0));   // max(0, 1) = 1
        REQUIRE(result(0, 1) == Catch::Approx(0.0));   // max(0, 0) = 0
        REQUIRE(result(0, 2) == Catch::Approx(0.0));   // max(0, -1) = 0
        REQUIRE(result(1, 0) == Catch::Approx(2.0));   // max(0, 2) = 2
        REQUIRE(result(1, 1) == Catch::Approx(0.0));   // max(0, -0.5) = 0
        REQUIRE(result(1, 2) == Catch::Approx(0.5));   // max(0, 0.5) = 0.5
    }
    
    SECTION("Softplus function") {
        Matrix result = softplus(x);
        
        REQUIRE(result.rows() == x.rows());
        REQUIRE(result.cols() == x.cols());
        
        // Check that all values are positive
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                REQUIRE(result(i, j) > 0.0);
            }
        }
        
        // Check specific values
        REQUIRE(result(0, 0) == Catch::Approx(std::log(1.0 + std::exp(1.0))));
        REQUIRE(result(0, 1) == Catch::Approx(std::log(2.0))); // softplus(0) = ln(2)
        REQUIRE(result(0, 2) == Catch::Approx(std::log(1.0 + std::exp(-1.0))));
    }
    
    SECTION("Softmax function") {
        Matrix small_x(1, 3);
        small_x << 1.0, 2.0, 3.0;
        
        Matrix result = softmax(small_x);
        
        REQUIRE(result.rows() == small_x.rows());
        REQUIRE(result.cols() == small_x.cols());
        
        // Check that row sums to 1
        Float row_sum = result.row(0).sum();
        REQUIRE(row_sum == Catch::Approx(1.0));
        
        // Check that all values are positive
        for (int j = 0; j < result.cols(); ++j) {
            REQUIRE(result(0, j) > 0.0);
        }
        
        // Check ordering (larger input -> larger output)
        REQUIRE(result(0, 0) < result(0, 1));
        REQUIRE(result(0, 1) < result(0, 2));
    }
    
    SECTION("Softmax with beta") {
        Matrix small_x(1, 3);
        small_x << 1.0, 2.0, 3.0;
        
        Matrix result_beta1 = softmax(small_x, 1.0);
        Matrix result_beta2 = softmax(small_x, 2.0);
        
        // With higher beta, the distribution should be more peaked
        Float entropy1 = 0.0, entropy2 = 0.0;
        for (int j = 0; j < result_beta1.cols(); ++j) {
            if (result_beta1(0, j) > 0) {
                entropy1 -= result_beta1(0, j) * std::log(result_beta1(0, j));
            }
            if (result_beta2(0, j) > 0) {
                entropy2 -= result_beta2(0, j) * std::log(result_beta2(0, j));
            }
        }
        
        REQUIRE(entropy2 < entropy1); // Higher beta -> lower entropy
    }
}

TEST_CASE("Activation registry", "[activations]") {
    auto& registry = ActivationRegistry::instance();
    
    SECTION("Get function by name") {
        auto identity_fn = registry.get_function("identity");
        auto sigmoid_fn = registry.get_function("sigmoid");
        auto tanh_fn = registry.get_function("tanh");
        auto relu_fn = registry.get_function("relu");
        auto softplus_fn = registry.get_function("softplus");
        auto softmax_fn = registry.get_function("softmax");
        
        Matrix x(1, 3);
        x << 1.0, 0.0, -1.0;
        
        // Test that functions work
        Matrix id_result = identity_fn(x);
        REQUIRE(id_result(0, 0) == Catch::Approx(1.0));
        REQUIRE(id_result(0, 1) == Catch::Approx(0.0));
        REQUIRE(id_result(0, 2) == Catch::Approx(-1.0));
        
        Matrix sig_result = sigmoid_fn(x);
        REQUIRE(sig_result(0, 1) == Catch::Approx(0.5));
        
        Matrix relu_result = relu_fn(x);
        REQUIRE(relu_result(0, 0) == Catch::Approx(1.0));
        REQUIRE(relu_result(0, 1) == Catch::Approx(0.0));
        REQUIRE(relu_result(0, 2) == Catch::Approx(0.0));
    }
    
    SECTION("Short names") {
        auto identity_fn = registry.get_function("id");
        auto sigmoid_fn = registry.get_function("sig");
        auto relu_fn = registry.get_function("re");
        auto softplus_fn = registry.get_function("sp");
        auto softmax_fn = registry.get_function("smax");
        
        Matrix x(1, 1);
        x << 1.0;
        
        // Just check that they work
        REQUIRE_NOTHROW(identity_fn(x));
        REQUIRE_NOTHROW(sigmoid_fn(x));
        REQUIRE_NOTHROW(relu_fn(x));
        REQUIRE_NOTHROW(softplus_fn(x));
        REQUIRE_NOTHROW(softmax_fn(x));
    }
    
    SECTION("Invalid function name") {
        REQUIRE_THROWS_AS(registry.get_function("invalid"), std::invalid_argument);
    }
    
    SECTION("Available functions") {
        auto names = registry.available_functions();
        REQUIRE(names.size() > 0);
        
        // Check that key functions are available
        bool has_identity = std::find(names.begin(), names.end(), "identity") != names.end();
        bool has_sigmoid = std::find(names.begin(), names.end(), "sigmoid") != names.end();
        bool has_tanh = std::find(names.begin(), names.end(), "tanh") != names.end();
        bool has_relu = std::find(names.begin(), names.end(), "relu") != names.end();
        
        REQUIRE(has_identity);
        REQUIRE(has_sigmoid);
        REQUIRE(has_tanh);
        REQUIRE(has_relu);
    }
}

TEST_CASE("Convenience function", "[activations]") {
    SECTION("get_function works") {
        auto identity_fn = get_function("identity");
        auto sigmoid_fn = get_function("sigmoid");
        
        Matrix x(1, 1);
        x << 2.0;
        
        Matrix id_result = identity_fn(x);
        Matrix sig_result = sigmoid_fn(x);
        
        REQUIRE(id_result(0, 0) == Catch::Approx(2.0));
        REQUIRE(sig_result(0, 0) == Catch::Approx(1.0 / (1.0 + std::exp(-2.0))));
    }
    
    SECTION("get_function throws for invalid name") {
        REQUIRE_THROWS_AS(get_function("invalid"), std::invalid_argument);
    }
}