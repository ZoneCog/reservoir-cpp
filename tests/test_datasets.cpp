/**
 * @file test_datasets.cpp
 * @brief Tests for datasets functionality
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "reservoircpp/datasets.hpp"
#include <cmath>

using namespace reservoircpp;
using namespace reservoircpp::datasets;

TEST_CASE("Datasets - Mackey-Glass", "[datasets]") {
    SECTION("Basic generation") {
        Matrix mg = mackey_glass(100);
        
        REQUIRE(mg.rows() == 100);
        REQUIRE(mg.cols() == 1);
        
        // Values should be positive and bounded
        REQUIRE(mg.minCoeff() > 0.0);
        REQUIRE(mg.maxCoeff() < 2.0);  // Typical MG range
    }
    
    SECTION("Parameter validation") {
        REQUIRE_THROWS_AS(mackey_glass(0), std::invalid_argument);
        REQUIRE_THROWS_AS(mackey_glass(100, 0), std::invalid_argument);
    }
    
    SECTION("Reproducibility") {
        Matrix mg1 = mackey_glass(50, 17, 0.2, 0.1, 10.0, 1.0, 1.2);
        Matrix mg2 = mackey_glass(50, 17, 0.2, 0.1, 10.0, 1.0, 1.2);
        
        REQUIRE(mg1.isApprox(mg2, 1e-10));
    }
}

TEST_CASE("Datasets - Lorenz", "[datasets]") {
    SECTION("Basic generation") {
        Matrix lorenz_data = lorenz(200);
        
        REQUIRE(lorenz_data.rows() == 200);
        REQUIRE(lorenz_data.cols() == 3);
        
        // Check that we have reasonable dynamics
        REQUIRE(lorenz_data.col(0).array().abs().maxCoeff() > 1.0);
        REQUIRE(lorenz_data.col(1).array().abs().maxCoeff() > 1.0);
        REQUIRE(lorenz_data.col(2).array().abs().maxCoeff() > 1.0);
    }
    
    SECTION("Parameter validation") {
        REQUIRE_THROWS_AS(lorenz(0), std::invalid_argument);
    }
    
    SECTION("Different parameters") {
        Matrix l1 = lorenz(100, 0.01, 10.0, 28.0, 8.0/3.0);
        Matrix l2 = lorenz(100, 0.01, 15.0, 28.0, 8.0/3.0);
        
        // Different sigma should give different results
        REQUIRE_FALSE(l1.isApprox(l2, 1e-3));
    }
}

TEST_CASE("Datasets - Hénon map", "[datasets]") {
    SECTION("Basic generation") {
        Matrix henon_data = henon_map(100);
        
        REQUIRE(henon_data.rows() == 100);
        REQUIRE(henon_data.cols() == 2);
        
        // Check typical Hénon map range
        REQUIRE(henon_data.col(0).array().abs().maxCoeff() < 2.0);
        REQUIRE(henon_data.col(1).array().abs().maxCoeff() < 1.0);
    }
    
    SECTION("Parameter validation") {
        REQUIRE_THROWS_AS(henon_map(0), std::invalid_argument);
    }
    
    SECTION("Classic parameters") {
        Matrix henon_classic = henon_map(50, 1.4, 0.3);
        
        REQUIRE(henon_classic.rows() == 50);
        REQUIRE(henon_classic.cols() == 2);
    }
}

TEST_CASE("Datasets - Logistic map", "[datasets]") {
    SECTION("Basic generation") {
        Matrix logistic_data = logistic_map(100);
        
        REQUIRE(logistic_data.rows() == 100);
        REQUIRE(logistic_data.cols() == 1);
        
        // Values should be in [0, 1]
        REQUIRE(logistic_data.minCoeff() >= 0.0);
        REQUIRE(logistic_data.maxCoeff() <= 1.0);
    }
    
    SECTION("Parameter validation") {
        REQUIRE_THROWS_AS(logistic_map(0), std::invalid_argument);
        REQUIRE_THROWS_AS(logistic_map(100, 4.0, 0.0), std::invalid_argument);
        REQUIRE_THROWS_AS(logistic_map(100, 4.0, 1.0), std::invalid_argument);
    }
    
    SECTION("Chaotic behavior") {
        Matrix chaotic = logistic_map(100, 3.9, 0.5, 10);  // Slightly lower r for stability
        
        // Check that we get non-zero values and some variation
        REQUIRE(chaotic.minCoeff() >= 0.0);
        REQUIRE(chaotic.maxCoeff() <= 1.0);
        REQUIRE(chaotic.maxCoeff() > 0.1);  // Should have some significant values
        
        // Check that there's some variation (not all the same value)
        Float variance = (chaotic.array() - chaotic.mean()).array().square().mean();
        REQUIRE(variance > 0.01);  // Should have some variation
    }
}

TEST_CASE("Datasets - NARMA", "[datasets]") {
    SECTION("Basic generation") {
        auto [input, target] = narma(100, 10);
        
        REQUIRE(input.rows() == 100);
        REQUIRE(input.cols() == 1);
        REQUIRE(target.rows() == 100);
        REQUIRE(target.cols() == 1);
        
        // Input should be in [0, 0.5]
        REQUIRE(input.minCoeff() >= 0.0);
        REQUIRE(input.maxCoeff() <= 0.5);
    }
    
    SECTION("Parameter validation") {
        REQUIRE_THROWS_AS(narma(0), std::invalid_argument);
        REQUIRE_THROWS_AS(narma(100, 0), std::invalid_argument);
    }
    
    SECTION("Different orders") {
        auto [input5, target5] = narma(50, 5);
        auto [input10, target10] = narma(50, 10);
        
        REQUIRE(input5.rows() == 50);
        REQUIRE(target5.rows() == 50);
        REQUIRE(input10.rows() == 50);
        REQUIRE(target10.rows() == 50);
    }
}

TEST_CASE("Datasets - To forecasting", "[datasets]") {
    SECTION("Basic forecasting") {
        Matrix ts(10, 2);
        for (int i = 0; i < 10; ++i) {
            ts(i, 0) = i;
            ts(i, 1) = i * 2;
        }
        
        auto [X, y] = to_forecasting(ts, 1);
        
        REQUIRE(X.rows() == 9);
        REQUIRE(X.cols() == 2);
        REQUIRE(y.rows() == 9);
        REQUIRE(y.cols() == 2);
        
        // Check that X[t] = ts[t] and y[t] = ts[t+1]
        REQUIRE(X(0, 0) == Catch::Approx(0.0));
        REQUIRE(y(0, 0) == Catch::Approx(1.0));
        REQUIRE(X(8, 0) == Catch::Approx(8.0));
        REQUIRE(y(8, 0) == Catch::Approx(9.0));
    }
    
    SECTION("Multi-step forecasting") {
        Matrix ts(10, 1);
        for (int i = 0; i < 10; ++i) {
            ts(i, 0) = i;
        }
        
        auto [X, y] = to_forecasting(ts, 3);
        
        REQUIRE(X.rows() == 7);
        REQUIRE(y.rows() == 7);
        
        // Check that y[t] = X[t+3]
        REQUIRE(X(0, 0) == Catch::Approx(0.0));
        REQUIRE(y(0, 0) == Catch::Approx(3.0));
    }
    
    SECTION("Parameter validation") {
        Matrix ts(5, 1);
        REQUIRE_THROWS_AS(to_forecasting(ts, 10), std::invalid_argument);
    }
}

TEST_CASE("Datasets - To forecasting with split", "[datasets]") {
    SECTION("Train-test split") {
        Matrix ts(20, 1);
        for (int i = 0; i < 20; ++i) {
            ts(i, 0) = i;
        }
        
        auto [X_train, X_test, y_train, y_test] = to_forecasting_with_split(ts, 1, 5);
        
        REQUIRE(X_train.rows() == 14);  // 19 - 5 = 14
        REQUIRE(X_test.rows() == 5);
        REQUIRE(y_train.rows() == 14);
        REQUIRE(y_test.rows() == 5);
        
        // Check continuity
        REQUIRE(X_test(0, 0) == Catch::Approx(14.0));
        REQUIRE(y_test(0, 0) == Catch::Approx(15.0));
    }
    
    SECTION("Parameter validation") {
        Matrix ts(10, 1);
        REQUIRE_THROWS_AS(to_forecasting_with_split(ts, 5, 10), std::invalid_argument);
    }
}

TEST_CASE("Datasets - One-hot encoding", "[datasets]") {
    SECTION("Basic encoding") {
        std::vector<int> labels = {0, 1, 2, 0, 1};
        Matrix encoded = one_hot_encode(labels, 3);
        
        REQUIRE(encoded.rows() == 5);
        REQUIRE(encoded.cols() == 3);
        
        // Check specific encodings
        REQUIRE(encoded(0, 0) == Catch::Approx(1.0));
        REQUIRE(encoded(0, 1) == Catch::Approx(0.0));
        REQUIRE(encoded(1, 1) == Catch::Approx(1.0));
        REQUIRE(encoded(2, 2) == Catch::Approx(1.0));
    }
    
    SECTION("Inferred classes") {
        std::vector<int> labels = {0, 2, 1, 2};
        Matrix encoded = one_hot_encode(labels);  // Should infer 3 classes
        
        REQUIRE(encoded.rows() == 4);
        REQUIRE(encoded.cols() == 3);
    }
    
    SECTION("Parameter validation") {
        std::vector<int> empty_labels;
        REQUIRE_THROWS_AS(one_hot_encode(empty_labels), std::invalid_argument);
        
        std::vector<int> invalid_labels = {0, 1, 5};
        REQUIRE_THROWS_AS(one_hot_encode(invalid_labels, 3), std::invalid_argument);
    }
}

TEST_CASE("Datasets - MSO (Multiple Superimposed Oscillators)", "[datasets]") {
    SECTION("Basic MSO") {
        std::vector<Float> freqs = {0.1, 0.2};
        Matrix mso_data = mso(100, freqs);
        
        REQUIRE(mso_data.rows() == 100);
        REQUIRE(mso_data.cols() == 1);
        
        // With normalization, should be in [-1, 1]
        REQUIRE(mso_data.minCoeff() >= -1.1);
        REQUIRE(mso_data.maxCoeff() <= 1.1);
    }
    
    SECTION("MSO without normalization") {
        std::vector<Float> freqs = {0.1};
        Matrix mso_data = mso(50, freqs, false);
        
        // Without normalization, should be sin wave in [-1, 1]
        REQUIRE(mso_data.minCoeff() >= -1.1);
        REQUIRE(mso_data.maxCoeff() <= 1.1);
    }
    
    SECTION("MSO2 (2 frequencies)") {
        Matrix mso2_data = mso2(100);
        
        REQUIRE(mso2_data.rows() == 100);
        REQUIRE(mso2_data.cols() == 1);
    }
    
    SECTION("MSO8 (8 frequencies)") {
        Matrix mso8_data = mso8(100);
        
        REQUIRE(mso8_data.rows() == 100);
        REQUIRE(mso8_data.cols() == 1);
    }
    
    SECTION("Parameter validation") {
        std::vector<Float> empty_freqs;
        REQUIRE_THROWS_AS(mso(100, empty_freqs), std::invalid_argument);
        REQUIRE_THROWS_AS(mso2(0), std::invalid_argument);
    }
}