/**
 * @file test_observables.cpp
 * @brief Tests for observables functionality
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "reservoircpp/observables.hpp"
#include "reservoircpp/matrix_generators.hpp"
#include <cmath>

using namespace reservoircpp;
using namespace reservoircpp::observables;

TEST_CASE("Observables - Array checking", "[observables]") {
    SECTION("Valid arrays") {
        Matrix y_true(5, 2);
        Matrix y_pred(5, 2);
        y_true.setRandom();
        y_pred.setRandom();
        
        REQUIRE_NOTHROW(check_arrays(y_true, y_pred));
    }
    
    SECTION("Mismatched rows") {
        Matrix y_true(5, 2);
        Matrix y_pred(4, 2);
        y_true.setRandom();
        y_pred.setRandom();
        
        REQUIRE_THROWS_AS(check_arrays(y_true, y_pred), std::invalid_argument);
    }
    
    SECTION("Mismatched columns") {
        Matrix y_true(5, 2);
        Matrix y_pred(5, 3);
        y_true.setRandom();
        y_pred.setRandom();
        
        REQUIRE_THROWS_AS(check_arrays(y_true, y_pred), std::invalid_argument);
    }
}

TEST_CASE("Observables - MSE", "[observables]") {
    SECTION("Perfect prediction") {
        Matrix y_true(3, 2);
        y_true << 1.0, 2.0,
                  3.0, 4.0,
                  5.0, 6.0;
        Matrix y_pred = y_true;
        
        REQUIRE(mse(y_true, y_pred) == Catch::Approx(0.0).margin(1e-10));
    }
    
    SECTION("Known MSE") {
        Matrix y_true(2, 1);
        Matrix y_pred(2, 1);
        y_true << 1.0, 3.0;
        y_pred << 2.0, 1.0;
        
        // MSE = ((1-2)² + (3-1)²) / 2 = (1 + 4) / 2 = 2.5
        REQUIRE(mse(y_true, y_pred) == Catch::Approx(2.5));
    }
}

TEST_CASE("Observables - RMSE", "[observables]") {
    SECTION("RMSE from MSE") {
        Matrix y_true(2, 1);
        Matrix y_pred(2, 1);
        y_true << 1.0, 3.0;
        y_pred << 2.0, 1.0;
        
        // RMSE = sqrt(MSE) = sqrt(2.5)
        REQUIRE(rmse(y_true, y_pred) == Catch::Approx(std::sqrt(2.5)));
    }
}

TEST_CASE("Observables - NRMSE", "[observables]") {
    SECTION("NRMSE with variance normalization") {
        Matrix y_true(4, 1);
        y_true << 1.0, 2.0, 3.0, 4.0;  // mean = 2.5, var = 1.25
        Matrix y_pred(4, 1);
        y_pred << 1.5, 2.5, 3.5, 4.5;  // offset by +0.5
        
        Float rmse_val = rmse(y_true, y_pred);
        Float var_y = (y_true.array() - y_true.mean()).array().square().mean();
        Float expected_nrmse = rmse_val / std::sqrt(var_y);
        
        REQUIRE(nrmse(y_true, y_pred, "var") == Catch::Approx(expected_nrmse));
    }
    
    SECTION("NRMSE with range normalization") {
        Matrix y_true(3, 1);
        y_true << 1.0, 3.0, 5.0;  // range = 4
        Matrix y_pred(3, 1);
        y_pred << 2.0, 3.0, 4.0;
        
        Float rmse_val = rmse(y_true, y_pred);
        Float range = y_true.maxCoeff() - y_true.minCoeff();
        Float expected_nrmse = rmse_val / range;
        
        REQUIRE(nrmse(y_true, y_pred, "range") == Catch::Approx(expected_nrmse));
    }
    
    SECTION("Invalid normalization") {
        Matrix y_true(2, 1);
        Matrix y_pred(2, 1);
        y_true.setRandom();
        y_pred.setRandom();
        
        REQUIRE_THROWS_AS(nrmse(y_true, y_pred, "invalid"), std::invalid_argument);
    }
}

TEST_CASE("Observables - R-squared", "[observables]") {
    SECTION("Perfect fit") {
        Matrix y_true(3, 1);
        y_true << 1.0, 2.0, 3.0;
        Matrix y_pred = y_true;
        
        REQUIRE(rsquare(y_true, y_pred) == Catch::Approx(1.0));
    }
    
    SECTION("No correlation (predict mean)") {
        Matrix y_true(3, 1);
        y_true << 1.0, 2.0, 3.0;  // mean = 2.0
        Matrix y_pred(3, 1);
        y_pred << 2.0, 2.0, 2.0;  // predict mean
        
        REQUIRE(rsquare(y_true, y_pred) == Catch::Approx(0.0));
    }
    
    SECTION("Known R²") {
        Matrix y_true(4, 1);
        y_true << 1.0, 2.0, 3.0, 4.0;  // mean = 2.5
        Matrix y_pred(4, 1);
        y_pred << 1.1, 1.9, 3.1, 3.9;
        
        // Calculate expected R²
        Float mean_y = y_true.mean();
        Float ss_res = (y_true - y_pred).array().square().sum();
        Float ss_tot = (y_true.array() - mean_y).array().square().sum();
        Float expected_r2 = 1.0 - (ss_res / ss_tot);
        
        REQUIRE(rsquare(y_true, y_pred) == Catch::Approx(expected_r2));
    }
    
    SECTION("Constant true values") {
        Matrix y_true(3, 1);
        y_true << 2.0, 2.0, 2.0;
        Matrix y_pred(3, 1);
        y_pred << 2.0, 2.0, 2.0;
        
        REQUIRE(rsquare(y_true, y_pred) == Catch::Approx(1.0));
    }
}

TEST_CASE("Observables - Spectral radius", "[observables]") {
    SECTION("Identity matrix") {
        Matrix I = Matrix::Identity(3, 3);
        REQUIRE(spectral_radius(I) == Catch::Approx(1.0));
    }
    
    SECTION("Zero matrix") {
        Matrix Z = Matrix::Zero(3, 3);
        REQUIRE(spectral_radius(Z) == Catch::Approx(0.0));
    }
    
    SECTION("Diagonal matrix") {
        Matrix D = Matrix::Zero(3, 3);
        D(0, 0) = 2.0;
        D(1, 1) = -3.0;
        D(2, 2) = 1.5;
        
        REQUIRE(spectral_radius(D) == Catch::Approx(3.0));  // max(|2|, |-3|, |1.5|) = 3
    }
    
    SECTION("Non-square matrix") {
        Matrix W(2, 3);
        W.setRandom();
        
        REQUIRE_THROWS_AS(spectral_radius(W), std::invalid_argument);
    }
    
    SECTION("Empty matrix") {
        Matrix W(0, 0);
        REQUIRE(spectral_radius(W) == Catch::Approx(0.0));
    }
}

TEST_CASE("Observables - Effective spectral radius", "[observables]") {
    SECTION("Stable dynamics") {
        Matrix states(10, 3);
        // Create decaying states
        for (int t = 0; t < 10; ++t) {
            states.row(t) = Vector::Random(3) * std::pow(0.8, t);
        }
        
        Float eff_sr = effective_spectral_radius(states);
        REQUIRE(eff_sr < 1.0);  // Should indicate stable dynamics
        REQUIRE(eff_sr > 0.0);
    }
    
    SECTION("Growing dynamics") {
        Matrix states(5, 2);
        // Create growing states
        for (int t = 0; t < 5; ++t) {
            states.row(t) = Vector::Ones(2) * std::pow(1.2, t);
        }
        
        Float eff_sr = effective_spectral_radius(states);
        REQUIRE(eff_sr > 1.0);  // Should indicate unstable dynamics
    }
    
    SECTION("Too few time steps") {
        Matrix states(1, 3);
        states.setRandom();
        
        REQUIRE_THROWS_AS(effective_spectral_radius(states), std::invalid_argument);
    }
}

TEST_CASE("Observables - Memory capacity", "[observables]") {
    SECTION("Perfect memory (identity transformation)") {
        int time_steps = 50;
        int delay = 5;
        
        // Create simple input signal
        Matrix input(time_steps, 1);
        for (int t = 0; t < time_steps; ++t) {
            input(t, 0) = std::sin(0.1 * t);
        }
        
        // Create reservoir states that perfectly remember past inputs
        Matrix states(time_steps, delay + 1);
        for (int t = 0; t < time_steps; ++t) {
            for (int d = 0; d <= delay; ++d) {
                if (t >= d) {
                    states(t, d) = input(t - d, 0);
                } else {
                    states(t, d) = 0.0;
                }
            }
        }
        
        Float mc = memory_capacity(states, input, delay);
        REQUIRE(mc > delay * 0.8);  // Should have high memory capacity
    }
    
    SECTION("No memory (random states)") {
        int time_steps = 30;
        Matrix input(time_steps, 1);
        Matrix states(time_steps, 10);
        
        input.setRandom();
        states.setRandom();
        
        Float mc = memory_capacity(states, input, 5);
        REQUIRE(mc >= 0.0);  // Should be non-negative
        REQUIRE(mc < 5.0);   // Should be less than max_delay for random states
    }
    
    SECTION("Mismatched dimensions") {
        Matrix states(10, 5);
        Matrix input(8, 1);  // Different number of time steps
        
        REQUIRE_THROWS_AS(memory_capacity(states, input, 3), std::invalid_argument);
    }
    
    SECTION("Multi-dimensional input") {
        Matrix states(10, 5);
        Matrix input(10, 2);  // 2D input
        
        REQUIRE_THROWS_AS(memory_capacity(states, input, 3), std::invalid_argument);
    }
    
    SECTION("Time series too short") {
        Matrix states(5, 3);
        Matrix input(5, 1);
        
        REQUIRE_THROWS_AS(memory_capacity(states, input, 10), std::invalid_argument);
    }
}