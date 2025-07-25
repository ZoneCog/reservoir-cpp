/**
 * @file test_reservoir.cpp
 * @brief Tests for reservoir classes
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <reservoircpp/reservoir.hpp>

using namespace reservoircpp;

TEST_CASE("Reservoir - Basic construction", "[reservoir]") {
    SECTION("Valid parameters") {
        Reservoir reservoir("test_reservoir", 100, 0.5);
        
        REQUIRE(reservoir.name() == "test_reservoir");
        REQUIRE(reservoir.units() == 100);
        REQUIRE_THAT(reservoir.leak_rate(), Catch::Matchers::WithinAbs(0.5, 1e-10));
        REQUIRE(reservoir.connectivity() == 0.1);  // Default value
        REQUIRE(reservoir.spectral_radius() == 0.9);  // Default value
        REQUIRE(reservoir.activation_name() == "tanh");  // Default value
    }
    
    SECTION("Invalid parameters") {
        REQUIRE_THROWS_AS(Reservoir("test", 0), std::invalid_argument);
        REQUIRE_THROWS_AS(Reservoir("test", 10, 0.0), std::invalid_argument);
        REQUIRE_THROWS_AS(Reservoir("test", 10, 1.5), std::invalid_argument);
        REQUIRE_THROWS_AS(Reservoir("test", 10, 0.5, "invalid_activation"), std::invalid_argument);
    }
}

TEST_CASE("Reservoir - Initialization", "[reservoir]") {
    SECTION("Initialize with input data") {
        Reservoir reservoir("test_reservoir", 20);
        
        Matrix x(10, 3);  // 10 samples, 3 features
        x.setRandom();
        
        reservoir.initialize(&x);
        
        REQUIRE(reservoir.is_reservoir_initialized());
        REQUIRE(reservoir.input_dim()[0] == 3);
        REQUIRE(reservoir.output_dim()[0] == 20);
        
        // Check weight matrices are initialized
        REQUIRE(reservoir.W().rows() == 20);
        REQUIRE(reservoir.W().cols() == 20);
        REQUIRE(reservoir.W_in().rows() == 20);
        REQUIRE(reservoir.W_in().cols() == 3);
    }
    
    SECTION("Initialize without input data") {
        Reservoir reservoir("test_reservoir", 15);
        
        REQUIRE_THROWS_AS(reservoir.initialize(), std::runtime_error);
    }
}

TEST_CASE("Reservoir - Forward pass", "[reservoir]") {
    SECTION("Single timestep") {
        Reservoir reservoir("test_reservoir", 10, 1.0);
        
        Matrix x(1, 2);  // 1 sample, 2 features
        x << 0.5, -0.3;
        
        reservoir.initialize(&x);
        
        Matrix output = reservoir.forward(x);
        
        REQUIRE(output.rows() == 1);
        REQUIRE(output.cols() == 10);
        
        // Output should be different from zero (reservoir should be active)
        REQUIRE(output.cwiseAbs().sum() > 0.01);
    }
    
    SECTION("Multiple timesteps") {
        Reservoir reservoir("test_reservoir", 5, 0.8);
        
        Matrix x(5, 2);  // 5 samples, 2 features
        x.setRandom();
        
        reservoir.initialize(&x);
        
        Matrix output = reservoir.forward(x);
        
        REQUIRE(output.rows() == 5);
        REQUIRE(output.cols() == 5);
        
        // Check that different timesteps produce different outputs
        REQUIRE_FALSE(output.row(0).isApprox(output.row(1), 1e-5));
    }
    
    SECTION("Dimension mismatch") {
        Reservoir reservoir("test_reservoir", 10);
        
        Matrix x_init(3, 2);
        x_init.setRandom();
        reservoir.initialize(&x_init);
        
        Matrix x_wrong(3, 3);  // Wrong input dimension
        x_wrong.setRandom();
        
        REQUIRE_THROWS_AS(reservoir.forward(x_wrong), std::invalid_argument);
    }
}

TEST_CASE("Reservoir - State management", "[reservoir]") {
    SECTION("Reset to zero") {
        Reservoir reservoir("test_reservoir", 8);
        
        Matrix x(2, 3);
        x.setRandom();
        
        reservoir.initialize(&x);
        
        // Run forward pass to change state
        reservoir.forward(x);
        
        // Reset and check state is zero
        reservoir.reset();
        Vector state = reservoir.get_state();
        
        REQUIRE(state.cwiseAbs().sum() < 1e-10);
    }
    
    SECTION("Reset to specific state") {
        Reservoir reservoir("test_reservoir", 6);
        
        Matrix x(2, 2);
        x.setRandom();
        
        reservoir.initialize(&x);
        
        Vector new_state(6);
        new_state.setOnes();
        
        reservoir.reset(&new_state);
        Vector state = reservoir.get_state();
        
        REQUIRE(state.isApprox(new_state, 1e-10));
    }
}

TEST_CASE("Reservoir - Copy functionality", "[reservoir]") {
    SECTION("Copy reservoir") {
        Reservoir reservoir("original", 12, 0.7, "sigmoid", 0.15, 0.85);
        
        Matrix x(3, 4);
        x.setRandom();
        
        reservoir.initialize(&x);
        
        auto copy = reservoir.copy("copy");
        auto copy_reservoir = dynamic_cast<Reservoir*>(copy.get());
        
        REQUIRE(copy_reservoir != nullptr);
        REQUIRE(copy_reservoir->name() == "copy");
        REQUIRE(copy_reservoir->units() == 12);
        REQUIRE(copy_reservoir->leak_rate() == 0.7);
        REQUIRE(copy_reservoir->activation_name() == "sigmoid");
        REQUIRE(copy_reservoir->connectivity() == 0.15);
        REQUIRE(copy_reservoir->spectral_radius() == 0.85);
    }
}

TEST_CASE("ESN - Specific functionality", "[esn]") {
    SECTION("ESN construction") {
        ESN esn("test_esn", 15, 0.9, 0.2, 0.95);
        
        REQUIRE(esn.name() == "test_esn");
        REQUIRE(esn.units() == 15);
        REQUIRE(esn.leak_rate() == 0.9);
        REQUIRE(esn.connectivity() == 0.2);
        REQUIRE(esn.spectral_radius() == 0.95);
        REQUIRE(esn.activation_name() == "tanh");  // ESN uses tanh by default
    }
    
    SECTION("ESN forward pass") {
        ESN esn("test_esn", 10);
        
        Matrix x(3, 2);
        x.setRandom();
        
        esn.initialize(&x);
        
        Matrix output = esn.forward(x);
        
        REQUIRE(output.rows() == 3);
        REQUIRE(output.cols() == 10);
        
        // Check output is within tanh range
        REQUIRE(output.cwiseAbs().maxCoeff() <= 1.0);
    }
    
    SECTION("ESN copy") {
        ESN esn("original_esn", 8);
        
        Matrix x(2, 3);
        x.setRandom();
        
        esn.initialize(&x);
        
        auto copy = esn.copy("copy_esn");
        auto copy_esn = dynamic_cast<ESN*>(copy.get());
        
        REQUIRE(copy_esn != nullptr);
        REQUIRE(copy_esn->name() == "copy_esn");
        REQUIRE(copy_esn->units() == 8);
    }
}

TEST_CASE("Reservoir - Parameter modifications", "[reservoir]") {
    SECTION("Modify leak rate") {
        Reservoir reservoir("test_reservoir", 10);
        
        reservoir.set_leak_rate(0.3);
        REQUIRE(reservoir.leak_rate() == 0.3);
    }
    
    SECTION("Modify connectivity") {
        Reservoir reservoir("test_reservoir", 10);
        
        reservoir.set_connectivity(0.25);
        REQUIRE(reservoir.connectivity() == 0.25);
    }
    
    SECTION("Modify spectral radius") {
        Reservoir reservoir("test_reservoir", 10);
        
        reservoir.set_spectral_radius(0.8);
        REQUIRE(reservoir.spectral_radius() == 0.8);
    }
}

TEST_CASE("Reservoir - Edge cases", "[reservoir]") {
    SECTION("Very small reservoir") {
        Reservoir reservoir("small", 1);
        
        Matrix x(1, 1);
        x << 1.0;
        
        reservoir.initialize(&x);
        
        Matrix output = reservoir.forward(x);
        
        REQUIRE(output.rows() == 1);
        REQUIRE(output.cols() == 1);
    }
    
    SECTION("High dimensional input") {
        Reservoir reservoir("high_dim", 5);
        
        Matrix x(2, 100);  // 100-dimensional input
        x.setRandom();
        
        reservoir.initialize(&x);
        
        Matrix output = reservoir.forward(x);
        
        REQUIRE(output.rows() == 2);
        REQUIRE(output.cols() == 5);
    }
}

TEST_CASE("IntrinsicPlasticity - Basic construction", "[intrinsic_plasticity]") {
    SECTION("Valid parameters with tanh") {
        IntrinsicPlasticity ip("test_ip", 50, 1.0, 0.0, 1.0, 5e-4, 1, "tanh");
        
        REQUIRE(ip.name() == "test_ip");
        REQUIRE(ip.units() == 50);
        REQUIRE_THAT(ip.leak_rate(), Catch::Matchers::WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(ip.mu(), Catch::Matchers::WithinAbs(0.0, 1e-10));
        REQUIRE_THAT(ip.sigma(), Catch::Matchers::WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(ip.learning_rate(), Catch::Matchers::WithinAbs(5e-4, 1e-10));
        REQUIRE(ip.epochs() == 1);
        REQUIRE(ip.activation_name() == "tanh");
    }
    
    SECTION("Valid parameters with sigmoid") {
        IntrinsicPlasticity ip("test_ip", 30, 0.8, 0.5, 0.5, 1e-3, 5, "sigmoid");
        
        REQUIRE(ip.name() == "test_ip");
        REQUIRE(ip.units() == 30);
        REQUIRE_THAT(ip.mu(), Catch::Matchers::WithinAbs(0.5, 1e-10));
        REQUIRE(ip.epochs() == 5);
        REQUIRE(ip.activation_name() == "sigmoid");
    }
    
    SECTION("Invalid parameters") {
        REQUIRE_THROWS_AS(IntrinsicPlasticity("test", 10, 1.0, 0.0, 1.0, 5e-4, 1, "relu"), 
                         std::invalid_argument);
        REQUIRE_THROWS_AS(IntrinsicPlasticity("test", 10, 1.0, 0.0, 1.0, -1e-4), 
                         std::invalid_argument);
        REQUIRE_THROWS_AS(IntrinsicPlasticity("test", 10, 1.0, 0.0, 1.0, 5e-4, 0), 
                         std::invalid_argument);
    }
}

TEST_CASE("IntrinsicPlasticity - Initialization", "[intrinsic_plasticity]") {
    SECTION("Initialize with input data") {
        IntrinsicPlasticity ip("test_ip", 20);
        
        Matrix x(10, 3);  // 10 samples, 3 features
        x.setRandom();
        
        ip.initialize(&x);
        
        REQUIRE(ip.is_reservoir_initialized());
        REQUIRE(ip.input_dim()[0] == 3);
        REQUIRE(ip.output_dim()[0] == 20);
        
        // Check IP parameters are initialized correctly
        REQUIRE(ip.a().rows() == 20);
        REQUIRE(ip.a().cols() == 1);
        REQUIRE(ip.b().rows() == 20);
        REQUIRE(ip.b().cols() == 1);
        
        // Initial a should be ones, b should be zeros
        REQUIRE(ip.a().isApprox(Matrix::Ones(20, 1), 1e-10));
        REQUIRE(ip.b().cwiseAbs().sum() < 1e-10);
    }
}

TEST_CASE("IntrinsicPlasticity - Forward pass", "[intrinsic_plasticity]") {
    SECTION("Forward pass produces valid output") {
        IntrinsicPlasticity ip("test_ip", 10, 1.0, 0.0, 1.0, 5e-4, 1, "tanh");
        
        Matrix x(5, 2);  // 5 samples, 2 features
        x.setRandom();
        
        ip.initialize(&x);
        
        Matrix output = ip.forward(x);
        
        REQUIRE(output.rows() == 5);
        REQUIRE(output.cols() == 10);
        
        // Output should be within tanh range
        REQUIRE(output.cwiseAbs().maxCoeff() <= 1.0);
    }
    
    SECTION("Forward pass with sigmoid activation") {
        IntrinsicPlasticity ip("test_ip", 8, 1.0, 0.5, 1.0, 1e-3, 1, "sigmoid");
        
        Matrix x(3, 2);
        x.setRandom();
        
        ip.initialize(&x);
        
        Matrix output = ip.forward(x);
        
        REQUIRE(output.rows() == 3);
        REQUIRE(output.cols() == 8);
        
        // Output should be in sigmoid range [0, 1]
        REQUIRE(output.minCoeff() >= 0.0);
        REQUIRE(output.maxCoeff() <= 1.0);
    }
}

TEST_CASE("IntrinsicPlasticity - Training functionality", "[intrinsic_plasticity]") {
    SECTION("Partial fit changes IP parameters") {
        IntrinsicPlasticity ip("test_ip", 5, 1.0, 0.0, 1.0, 1e-2, 1, "tanh");
        
        Matrix x(20, 2);  // Long sequence for training
        x.setRandom();
        
        ip.initialize(&x);
        
        // Store initial parameters
        Matrix a_initial = ip.a();
        Matrix b_initial = ip.b();
        
        // Train with partial fit
        ip.partial_fit(x, 5);  // 5 warmup steps
        
        // Parameters should have changed
        REQUIRE_FALSE(ip.a().isApprox(a_initial, 1e-3));
        REQUIRE_FALSE(ip.b().isApprox(b_initial, 1e-3));
    }
    
    SECTION("Fit with multiple sequences") {
        IntrinsicPlasticity ip("test_ip", 8, 1.0, 0.0, 1.0, 5e-3, 2, "tanh");
        
        Matrix x1(15, 3);
        Matrix x2(15, 3);
        x1.setRandom();
        x2.setRandom();
        
        std::vector<Matrix> sequences = {x1, x2};
        
        ip.initialize(&x1);
        
        // Store initial parameters
        Matrix a_initial = ip.a();
        Matrix b_initial = ip.b();
        
        // Train with multiple sequences and epochs
        ip.fit(sequences, 3);  // 3 warmup steps
        
        // Parameters should have changed after training
        REQUIRE_FALSE(ip.a().isApprox(a_initial, 1e-3));
        REQUIRE_FALSE(ip.b().isApprox(b_initial, 1e-3));
    }
}

TEST_CASE("IntrinsicPlasticity - Copy functionality", "[intrinsic_plasticity]") {
    SECTION("Copy IP reservoir") {
        IntrinsicPlasticity ip("original", 12, 0.8, 0.2, 0.8, 1e-3, 3, "sigmoid");
        
        Matrix x(5, 4);
        x.setRandom();
        
        ip.initialize(&x);
        
        auto copy = ip.copy("copy");
        auto copy_ip = dynamic_cast<IntrinsicPlasticity*>(copy.get());
        
        REQUIRE(copy_ip != nullptr);
        REQUIRE(copy_ip->name() == "copy");
        REQUIRE(copy_ip->units() == 12);
        REQUIRE(copy_ip->leak_rate() == 0.8);
        REQUIRE(copy_ip->mu() == 0.2);
        REQUIRE(copy_ip->sigma() == 0.8);
        REQUIRE(copy_ip->learning_rate() == 1e-3);
        REQUIRE(copy_ip->epochs() == 3);
        REQUIRE(copy_ip->activation_name() == "sigmoid");
        
        // Check that IP parameters are copied
        REQUIRE(copy_ip->a().isApprox(ip.a(), 1e-10));
        REQUIRE(copy_ip->b().isApprox(ip.b(), 1e-10));
    }
}

TEST_CASE("IntrinsicPlasticity - Parameter behavior", "[intrinsic_plasticity]") {
    SECTION("IP parameters evolve during training") {
        IntrinsicPlasticity ip("test_ip", 3, 1.0, 0.0, 1.0, 0.1, 1, "tanh");  // High learning rate for testing
        
        Matrix x(10, 1);
        x.setConstant(0.5);  // Constant input
        
        ip.initialize(&x);
        
        // Store initial values
        Matrix a_before = ip.a();
        Matrix b_before = ip.b();
        
        // Train
        ip.partial_fit(x, 2);
        
        // Values should change
        Matrix a_after = ip.a();
        Matrix b_after = ip.b();
        
        bool a_changed = !a_after.isApprox(a_before, 1e-6);
        bool b_changed = !b_after.isApprox(b_before, 1e-6);
        
        REQUIRE((a_changed || b_changed));  // At least one parameter should change
    }
}

TEST_CASE("NVAR - Basic construction", "[nvar]") {
    SECTION("Valid parameters") {
        NVAR nvar("test_nvar", 3, 2, 1);
        
        REQUIRE(nvar.name() == "test_nvar");
        REQUIRE(nvar.delay() == 3);
        REQUIRE(nvar.order() == 2);
        REQUIRE(nvar.strides() == 1);
    }
    
    SECTION("With strides") {
        NVAR nvar("test_nvar", 5, 3, 2);
        
        REQUIRE(nvar.delay() == 5);
        REQUIRE(nvar.order() == 3);
        REQUIRE(nvar.strides() == 2);
    }
    
    SECTION("Invalid parameters") {
        REQUIRE_THROWS_AS(NVAR("test", 0, 2), std::invalid_argument);
        REQUIRE_THROWS_AS(NVAR("test", 3, 0), std::invalid_argument);
        REQUIRE_THROWS_AS(NVAR("test", 3, 2, 0), std::invalid_argument);
    }
}

TEST_CASE("NVAR - Initialization", "[nvar]") {
    SECTION("Initialize with input data") {
        NVAR nvar("test_nvar", 2, 2);  // delay=2, order=2
        
        Matrix x(5, 3);  // 5 samples, 3 features
        x.setRandom();
        
        nvar.initialize(&x);
        
        REQUIRE(nvar.input_dim()[0] == 3);
        
        // Linear dimension: delay * input_dim = 2 * 3 = 6
        REQUIRE(nvar.linear_dim() == 6);
        
        // Nonlinear dimension: C(6+2-1, 2) = C(7, 2) = 21
        REQUIRE(nvar.nonlinear_dim() == 21);
        
        // Total output dimension: 6 + 21 = 27
        REQUIRE(nvar.output_dim()[0] == 27);
        
        // Storage should be initialized
        REQUIRE(nvar.store().rows() == 2);  // delay * strides = 2 * 1
        REQUIRE(nvar.store().cols() == 3);  // input_dim
    }
    
    SECTION("Initialize with strides") {
        NVAR nvar("test_nvar", 3, 2, 2);  // delay=3, order=2, strides=2
        
        Matrix x(5, 2);  // 5 samples, 2 features
        x.setRandom();
        
        nvar.initialize(&x);
        
        // Linear dimension: delay * input_dim = 3 * 2 = 6
        REQUIRE(nvar.linear_dim() == 6);
        
        // Storage rows: delay * strides = 3 * 2 = 6
        REQUIRE(nvar.store().rows() == 6);
        REQUIRE(nvar.store().cols() == 2);
    }
    
    SECTION("Initialize without input data") {
        NVAR nvar("test_nvar", 2, 2);
        
        REQUIRE_THROWS_AS(nvar.initialize(), std::runtime_error);
    }
}

TEST_CASE("NVAR - Forward pass", "[nvar]") {
    SECTION("Single timestep") {
        NVAR nvar("test_nvar", 2, 2);  // delay=2, order=2
        
        Matrix x(1, 2);  // 1 sample, 2 features
        x << 0.5, -0.3;
        
        nvar.initialize(&x);
        
        Matrix output = nvar.forward(x);
        
        REQUIRE(output.rows() == 1);
        REQUIRE(output.cols() == 14);  // 4 linear + 10 nonlinear features
        
        // First timestep with no history should have zeros for delayed features
        // but current input should be present
        REQUIRE(output(0, 0) == 0.5);   // First linear feature (current input)
        REQUIRE(output(0, 1) == -0.3);  // Second linear feature (current input)
        REQUIRE(output(0, 2) == 0.0);   // Third linear feature (t-1, should be 0)
        REQUIRE(output(0, 3) == 0.0);   // Fourth linear feature (t-1, should be 0)
    }
    
    SECTION("Multiple timesteps build history") {
        NVAR nvar("test_nvar", 2, 2);
        
        Matrix x(3, 2);
        x << 1.0, 2.0,   // t=0
             3.0, 4.0,   // t=1
             5.0, 6.0;   // t=2
        
        nvar.initialize(&x);
        
        Matrix output = nvar.forward(x);
        
        REQUIRE(output.rows() == 3);
        REQUIRE(output.cols() == 14);
        
        // At t=1, we should have history from t=0
        REQUIRE(output(1, 0) == 3.0);  // Current input feature 1
        REQUIRE(output(1, 1) == 4.0);  // Current input feature 2
        REQUIRE(output(1, 2) == 1.0);  // t-1 input feature 1
        REQUIRE(output(1, 3) == 2.0);  // t-1 input feature 2
        
        // At t=2, we should have full history
        REQUIRE(output(2, 0) == 5.0);  // Current input feature 1
        REQUIRE(output(2, 1) == 6.0);  // Current input feature 2
        REQUIRE(output(2, 2) == 3.0);  // t-1 input feature 1
        REQUIRE(output(2, 3) == 4.0);  // t-1 input feature 2
    }
    
    SECTION("Nonlinear features are computed") {
        NVAR nvar("test_nvar", 1, 2);  // Simple case: delay=1, order=2
        
        Matrix x(1, 2);
        x << 2.0, 3.0;
        
        nvar.initialize(&x);
        
        Matrix output = nvar.forward(x);
        
        // Linear features: [2.0, 3.0]
        REQUIRE(output(0, 0) == 2.0);
        REQUIRE(output(0, 1) == 3.0);
        
        // Nonlinear features (monomials of order 2):
        // x1*x1, x1*x2, x2*x2 = 4.0, 6.0, 9.0
        REQUIRE(output(0, 2) == 4.0);  // 2.0 * 2.0
        REQUIRE(output(0, 3) == 6.0);  // 2.0 * 3.0
        REQUIRE(output(0, 4) == 9.0);  // 3.0 * 3.0
    }
    
    SECTION("Dimension mismatch") {
        NVAR nvar("test_nvar", 2, 2);
        
        Matrix x_init(3, 2);
        x_init.setRandom();
        nvar.initialize(&x_init);
        
        Matrix x_wrong(3, 3);  // Wrong input dimension
        x_wrong.setRandom();
        
        REQUIRE_THROWS_AS(nvar.forward(x_wrong), std::invalid_argument);
    }
}

TEST_CASE("NVAR - Copy functionality", "[nvar]") {
    SECTION("Copy NVAR node") {
        NVAR nvar("original", 3, 2, 2);
        
        Matrix x(5, 4);
        x.setRandom();
        
        nvar.initialize(&x);
        
        auto copy = nvar.copy("copy");
        auto copy_nvar = dynamic_cast<NVAR*>(copy.get());
        
        REQUIRE(copy_nvar != nullptr);
        REQUIRE(copy_nvar->name() == "copy");
        REQUIRE(copy_nvar->delay() == 3);
        REQUIRE(copy_nvar->order() == 2);
        REQUIRE(copy_nvar->strides() == 2);
        REQUIRE(copy_nvar->linear_dim() == nvar.linear_dim());
        REQUIRE(copy_nvar->nonlinear_dim() == nvar.nonlinear_dim());
        REQUIRE(copy_nvar->input_dim() == nvar.input_dim());
        REQUIRE(copy_nvar->output_dim() == nvar.output_dim());
    }
}

TEST_CASE("NVAR - Reset functionality", "[nvar]") {
    SECTION("Reset clears history") {
        NVAR nvar("test_nvar", 2, 2);
        
        Matrix x(3, 2);
        x.setOnes();
        
        nvar.initialize(&x);
        
        // Run forward to build history
        nvar.forward(x);
        
        // Reset should clear the storage
        nvar.reset();
        
        // After reset, storage should be zero
        REQUIRE(nvar.store().cwiseAbs().sum() < 1e-10);
    }
}

TEST_CASE("NVAR - Strides functionality", "[nvar]") {
    SECTION("Strides affect delay sampling") {
        NVAR nvar("test_nvar", 2, 2, 2);  // delay=2, strides=2
        
        Matrix x(5, 1);
        x << 1.0, 2.0, 3.0, 4.0, 5.0;
        
        nvar.initialize(&x);
        
        Matrix output = nvar.forward(x);
        
        // With strides=2, we sample every 2nd timestep for delays
        // So at t=4 (5th sample), we use current (5.0) and t-2 (3.0), t-4 (1.0)
        REQUIRE(output(4, 0) == 5.0);  // Current
        REQUIRE(output(4, 1) == 3.0);  // t-2 (due to strides=2)
    }
}