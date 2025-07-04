/**
 * @file test_readout.cpp
 * @brief Tests for readout classes
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <reservoircpp/readout.hpp>

using namespace reservoircpp;

TEST_CASE("RidgeReadout - Basic functionality", "[readout][ridge]") {
    SECTION("Construction") {
        RidgeReadout readout("test_ridge", 2, 1e-6);
        
        REQUIRE(readout.name() == "test_ridge");
        REQUIRE(readout.output_dim()[0] == 2);
        REQUIRE(readout.ridge() == 1e-6);
        REQUIRE(readout.input_bias() == true);
        REQUIRE(readout.is_trainable() == true);
        REQUIRE(readout.is_fitted() == false);
    }
    
    SECTION("Invalid parameters") {
        REQUIRE_THROWS_AS(RidgeReadout("test", 0), std::invalid_argument);
        REQUIRE_THROWS_AS(RidgeReadout("test", 2, -1.0), std::invalid_argument);
    }
}

TEST_CASE("RidgeReadout - Training and prediction", "[readout][ridge]") {
    SECTION("Simple regression") {
        RidgeReadout readout("test_ridge", 1, 1e-8);
        
        // Create simple training data: y = 2*x1 + 3*x2 + 1
        Matrix x(10, 2);
        Matrix y(10, 1);
        
        for (int i = 0; i < 10; ++i) {
            x(i, 0) = i * 0.1;
            x(i, 1) = (i + 1) * 0.2;
            y(i, 0) = 2.0 * x(i, 0) + 3.0 * x(i, 1) + 1.0;
        }
        
        readout.fit(x, y);
        
        REQUIRE(readout.is_fitted() == true);
        
        // Test prediction
        Matrix x_test(3, 2);
        x_test << 0.1, 0.2,
                  0.3, 0.4,
                  0.5, 0.6;
        
        Matrix y_pred = readout.predict(x_test);
        
        REQUIRE(y_pred.rows() == 3);
        REQUIRE(y_pred.cols() == 1);
        
        // Check predictions are reasonable
        Matrix y_true(3, 1);
        y_true << 2.0 * 0.1 + 3.0 * 0.2 + 1.0,
                  2.0 * 0.3 + 3.0 * 0.4 + 1.0,
                  2.0 * 0.5 + 3.0 * 0.6 + 1.0;
        
        REQUIRE(y_pred.isApprox(y_true, 0.1));
    }
    
    SECTION("Multiple outputs") {
        RidgeReadout readout("test_ridge", 2, 1e-6);
        
        Matrix x(8, 3);
        Matrix y(8, 2);
        x.setRandom();
        y.setRandom();
        
        readout.fit(x, y);
        
        REQUIRE(readout.is_fitted() == true);
        
        Matrix y_pred = readout.predict(x);
        
        REQUIRE(y_pred.rows() == 8);
        REQUIRE(y_pred.cols() == 2);
    }
    
    SECTION("Dimension mismatch") {
        RidgeReadout readout("test_ridge", 1);
        
        Matrix x(5, 2);
        Matrix y(3, 1);  // Wrong number of samples
        x.setRandom();
        y.setRandom();
        
        REQUIRE_THROWS_AS(readout.fit(x, y), std::invalid_argument);
    }
}

TEST_CASE("RidgeReadout - Copy functionality", "[readout][ridge]") {
    SECTION("Copy readout") {
        RidgeReadout readout("original", 2, 1e-5);
        
        Matrix x(5, 3);
        Matrix y(5, 2);
        x.setRandom();
        y.setRandom();
        
        readout.fit(x, y);
        
        auto copy = readout.copy("copy");
        auto copy_readout = dynamic_cast<RidgeReadout*>(copy.get());
        
        REQUIRE(copy_readout != nullptr);
        REQUIRE(copy_readout->name() == "copy");
        REQUIRE(copy_readout->ridge() == 1e-5);
        REQUIRE(copy_readout->is_fitted() == true);
        
        // Check predictions are the same
        Matrix y_pred1 = readout.predict(x);
        Matrix y_pred2 = copy_readout->predict(x);
        
        REQUIRE(y_pred1.isApprox(y_pred2, 1e-10));
    }
}

TEST_CASE("ForceReadout - Basic functionality", "[readout][force]") {
    SECTION("Construction") {
        ForceReadout readout("test_force", 1, 1.0, 1.0);
        
        REQUIRE(readout.name() == "test_force");
        REQUIRE(readout.output_dim()[0] == 1);
        REQUIRE(readout.learning_rate() == 1.0);
        REQUIRE(readout.regularization() == 1.0);
        REQUIRE(readout.is_fitted() == false);
    }
    
    SECTION("Invalid parameters") {
        REQUIRE_THROWS_AS(ForceReadout("test", 1, 0.0), std::invalid_argument);
        REQUIRE_THROWS_AS(ForceReadout("test", 1, 1.5), std::invalid_argument);
        REQUIRE_THROWS_AS(ForceReadout("test", 1, 1.0, -1.0), std::invalid_argument);
    }
}

TEST_CASE("ForceReadout - Training", "[readout][force]") {
    SECTION("Batch training") {
        ForceReadout readout("test_force", 1, 1.0, 1.0);
        
        Matrix x(10, 2);
        Matrix y(10, 1);
        x.setRandom();
        y.setRandom();
        
        readout.fit(x, y);
        
        REQUIRE(readout.is_fitted() == true);
        
        Matrix y_pred = readout.predict(x);
        
        REQUIRE(y_pred.rows() == 10);
        REQUIRE(y_pred.cols() == 1);
    }
    
    SECTION("Online training") {
        ForceReadout readout("test_force", 1, 1.0, 1.0);
        
        Matrix x_init(1, 2);
        Matrix y_init(1, 1);
        x_init.setRandom();
        y_init.setRandom();
        
        readout.initialize(&x_init, &y_init);
        
        // Train one sample at a time
        for (int i = 0; i < 5; ++i) {
            Matrix x_sample(1, 2);
            Matrix y_sample(1, 1);
            x_sample.setRandom();
            y_sample.setRandom();
            
            readout.partial_fit(x_sample, y_sample);
        }
        
        REQUIRE(readout.is_fitted() == true);
    }
}

TEST_CASE("LMSReadout - Basic functionality", "[readout][lms]") {
    SECTION("Construction") {
        LMSReadout readout("test_lms", 1, 0.01);
        
        REQUIRE(readout.name() == "test_lms");
        REQUIRE(readout.output_dim()[0] == 1);
        REQUIRE(readout.learning_rate() == 0.01);
        REQUIRE(readout.is_fitted() == false);
    }
    
    SECTION("Invalid parameters") {
        REQUIRE_THROWS_AS(LMSReadout("test", 1, -0.01), std::invalid_argument);
    }
}

TEST_CASE("LMSReadout - Training", "[readout][lms]") {
    SECTION("Batch training") {
        LMSReadout readout("test_lms", 1, 0.1);
        
        Matrix x(20, 3);
        Matrix y(20, 1);
        x.setRandom();
        y.setRandom();
        
        readout.fit(x, y);
        
        REQUIRE(readout.is_fitted() == true);
        
        Matrix y_pred = readout.predict(x);
        
        REQUIRE(y_pred.rows() == 20);
        REQUIRE(y_pred.cols() == 1);
    }
    
    SECTION("Online training") {
        LMSReadout readout("test_lms", 1, 0.1);
        
        Matrix x_init(1, 3);
        Matrix y_init(1, 1);
        x_init.setRandom();
        y_init.setRandom();
        
        readout.initialize(&x_init, &y_init);
        
        // Train one sample at a time
        for (int i = 0; i < 10; ++i) {
            Matrix x_sample(1, 3);
            Matrix y_sample(1, 1);
            x_sample.setRandom();
            y_sample.setRandom();
            
            readout.partial_fit(x_sample, y_sample);
        }
        
        REQUIRE(readout.is_fitted() == true);
    }
}

TEST_CASE("Readout - Input bias handling", "[readout]") {
    SECTION("With bias") {
        RidgeReadout readout("test_bias", 1, 1e-8, true);
        
        Matrix x(5, 2);
        Matrix y(5, 1);
        x.setRandom();
        y.setRandom();
        
        readout.fit(x, y);
        
        // Check that input dimension includes bias
        REQUIRE(readout.input_dim()[0] == 3);  // 2 features + 1 bias
    }
    
    SECTION("Without bias") {
        RidgeReadout readout("test_no_bias", 1, 1e-8, false);
        
        Matrix x(5, 2);
        Matrix y(5, 1);
        x.setRandom();
        y.setRandom();
        
        readout.fit(x, y);
        
        // Check that input dimension doesn't include bias
        REQUIRE(readout.input_dim()[0] == 2);  // 2 features only
    }
}

TEST_CASE("Readout - Error handling", "[readout]") {
    SECTION("Prediction without fitting") {
        RidgeReadout readout("test_ridge", 1);
        
        Matrix x(3, 2);
        x.setRandom();
        
        REQUIRE_THROWS_AS(readout.predict(x), std::runtime_error);
        REQUIRE_THROWS_AS(readout.forward(x), std::runtime_error);
    }
    
    SECTION("Partial fit parameter validation") {
        ForceReadout readout("test_force", 1);
        
        Matrix x(2, 2);  // Multiple samples
        Matrix y(2, 1);
        x.setRandom();
        y.setRandom();
        
        REQUIRE_THROWS_AS(readout.partial_fit(x, y), std::invalid_argument);
    }
}

TEST_CASE("Readout - Reset functionality", "[readout]") {
    SECTION("Reset after training") {
        RidgeReadout readout("test_ridge", 1);
        
        Matrix x(5, 2);
        Matrix y(5, 1);
        x.setRandom();
        y.setRandom();
        
        readout.fit(x, y);
        REQUIRE(readout.is_fitted() == true);
        
        readout.reset();
        REQUIRE(readout.is_fitted() == false);
        
        // Should throw when trying to predict after reset
        REQUIRE_THROWS_AS(readout.predict(x), std::runtime_error);
    }
}