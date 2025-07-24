/**
 * @file test_matrix_generators.cpp
 * @brief Tests for matrix generators
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <reservoircpp/matrix_generators.hpp>

using namespace reservoircpp;
using namespace reservoircpp::matrix_generators;

TEST_CASE("Matrix generators - Basic functionality", "[matrix_generators]") {
    SECTION("Uniform distribution") {
        Matrix m = uniform(3, 4, -1.0, 1.0);
        REQUIRE(m.rows() == 3);
        REQUIRE(m.cols() == 4);
        
        // Check values are within bounds
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                REQUIRE(m(i, j) >= -1.0);
                REQUIRE(m(i, j) <= 1.0);
            }
        }
    }
    
    SECTION("Normal distribution") {
        Matrix m = normal(5, 3, 0.0, 1.0);
        REQUIRE(m.rows() == 5);
        REQUIRE(m.cols() == 3);
        
        // Check basic statistics (should be close to N(0,1))
        Float mean = m.mean();
        REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0.0, 0.5));
    }
    
    SECTION("Bernoulli distribution") {
        Matrix m = bernoulli(4, 4, 0.5);
        REQUIRE(m.rows() == 4);
        REQUIRE(m.cols() == 4);
        
        // Check values are only -1 or 1
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                REQUIRE((m(i, j) == -1.0 || m(i, j) == 1.0));
            }
        }
    }
    
    SECTION("Zeros and ones") {
        Matrix z = zeros(3, 3);
        Matrix o = ones(2, 5);
        
        REQUIRE(z.rows() == 3);
        REQUIRE(z.cols() == 3);
        REQUIRE(o.rows() == 2);
        REQUIRE(o.cols() == 5);
        
        REQUIRE(z.sum() == 0.0);
        REQUIRE(o.sum() == 10.0);
    }
}

TEST_CASE("Matrix generators - Connectivity", "[matrix_generators]") {
    SECTION("Sparse matrix generation") {
        SparseMatrix sm = random_sparse(10, 10, 0.1, "uniform");
        REQUIRE(sm.rows() == 10);
        REQUIRE(sm.cols() == 10);
        
        // Check sparsity (should be approximately 10% non-zero)
        int nnz = sm.nonZeros();
        REQUIRE(nnz > 0);
        REQUIRE(nnz < 50);  // Should be much less than 100 elements
    }
    
    SECTION("Dense matrix with connectivity") {
        Matrix m = uniform(5, 5, -1.0, 1.0, 0.3);
        REQUIRE(m.rows() == 5);
        REQUIRE(m.cols() == 5);
        
        // Count non-zero elements
        int nnz = 0;
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                if (std::abs(m(i, j)) > 1e-10) {
                    nnz++;
                }
            }
        }
        
        // Should have some zeros due to connectivity < 1
        REQUIRE(nnz < 25);
    }
}

TEST_CASE("Matrix generators - Spectral radius", "[matrix_generators]") {
    SECTION("Spectral radius computation") {
        Matrix m = Matrix::Identity(3, 3);
        Float sr = spectral_radius(m);
        REQUIRE_THAT(sr, Catch::Matchers::WithinAbs(1.0, 1e-10));
    }
    
    SECTION("Spectral radius scaling") {
        Matrix m = uniform(5, 5, -1.0, 1.0);
        Float target_sr = 0.9;
        Matrix scaled = scale_spectral_radius(m, target_sr);
        
        Float actual_sr = spectral_radius(scaled);
        REQUIRE_THAT(actual_sr, Catch::Matchers::WithinAbs(target_sr, 0.1));
    }
}

TEST_CASE("Matrix generators - Reservoir weights", "[matrix_generators]") {
    SECTION("Internal weights generation") {
        Matrix W = generate_internal_weights(10, 0.5, 0.9);
        REQUIRE(W.rows() == 10);
        REQUIRE(W.cols() == 10);
        
        Float sr = spectral_radius(W);
        REQUIRE_THAT(sr, Catch::Matchers::WithinAbs(0.9, 0.1));
    }
    
    SECTION("Input weights generation") {
        Matrix W_in = generate_input_weights(20, 5, 1.0);
        REQUIRE(W_in.rows() == 20);
        REQUIRE(W_in.cols() == 5);
        
        // Check scaling
        Float max_val = W_in.cwiseAbs().maxCoeff();
        REQUIRE(max_val > 0.5);  // Should have some significant values
    }
}

TEST_CASE("Matrix generators - Error handling", "[matrix_generators]") {
    SECTION("Invalid parameters") {
        REQUIRE_THROWS_AS(uniform(3, 3, 1.0, -1.0), std::invalid_argument);
        REQUIRE_THROWS_AS(normal(3, 3, 0.0, -1.0), std::invalid_argument);
        REQUIRE_THROWS_AS(bernoulli(3, 3, 1.5), std::invalid_argument);
        REQUIRE_THROWS_AS(random_sparse(3, 3, 1.5), std::invalid_argument);
    }
    
    SECTION("Invalid matrix for spectral radius") {
        Matrix m(3, 4);  // Non-square matrix
        REQUIRE_THROWS_AS(spectral_radius(m), std::invalid_argument);
    }
}

TEST_CASE("Matrix generators - Reproducibility", "[matrix_generators]") {
    SECTION("Same seed produces same result") {
        Matrix m1 = uniform(3, 3, -1.0, 1.0, 1.0, 42);
        Matrix m2 = uniform(3, 3, -1.0, 1.0, 1.0, 42);
        
        REQUIRE(m1.isApprox(m2, 1e-10));
    }
    
    SECTION("Different seed produces different result") {
        Matrix m1 = uniform(3, 3, -1.0, 1.0, 1.0, 42);
        Matrix m2 = uniform(3, 3, -1.0, 1.0, 1.0, 43);
        
        REQUIRE_FALSE(m1.isApprox(m2, 1e-5));
    }
}