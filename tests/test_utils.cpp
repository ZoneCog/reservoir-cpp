/**
 * @file test_utils.cpp
 * @brief Tests for utility functions
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <reservoircpp/utils.hpp>

using namespace reservoircpp;
using namespace reservoircpp::utils;

TEST_CASE("Random number generation", "[utils]") {
    SECTION("Random generator singleton") {
        auto& gen1 = RandomGenerator::instance();
        auto& gen2 = RandomGenerator::instance();
        
        REQUIRE(&gen1 == &gen2); // Same instance
    }
    
    SECTION("Set seed reproducibility") {
        auto& gen = RandomGenerator::instance();
        
        gen.set_seed(42);
        Float val1 = gen.uniform();
        Float val2 = gen.uniform();
        
        gen.set_seed(42);
        Float val3 = gen.uniform();
        Float val4 = gen.uniform();
        
        REQUIRE(val1 == Catch::Approx(val3));
        REQUIRE(val2 == Catch::Approx(val4));
    }
    
    SECTION("Uniform distribution") {
        auto& gen = RandomGenerator::instance();
        
        // Test default range [0, 1)
        for (int i = 0; i < 100; ++i) {
            Float val = gen.uniform();
            REQUIRE(val >= 0.0);
            REQUIRE(val < 1.0);
        }
        
        // Test custom range
        for (int i = 0; i < 100; ++i) {
            Float val = gen.uniform(-5.0, 5.0);
            REQUIRE(val >= -5.0);
            REQUIRE(val < 5.0);
        }
    }
    
    SECTION("Normal distribution") {
        auto& gen = RandomGenerator::instance();
        
        // Generate many samples and check approximate statistics
        std::vector<Float> samples;
        for (int i = 0; i < 1000; ++i) {
            samples.push_back(gen.normal(0.0, 1.0));
        }
        
        // Calculate mean and std dev
        Float sum = 0.0;
        for (Float val : samples) {
            sum += val;
        }
        Float mean = sum / samples.size();
        
        Float var_sum = 0.0;
        for (Float val : samples) {
            var_sum += (val - mean) * (val - mean);
        }
        Float std_dev = std::sqrt(var_sum / samples.size());
        
        // Check that they are approximately correct
        REQUIRE(mean == Catch::Approx(0.0).margin(0.1));
        REQUIRE(std_dev == Catch::Approx(1.0).margin(0.1));
    }
    
    SECTION("Random integers") {
        auto& gen = RandomGenerator::instance();
        
        for (int i = 0; i < 100; ++i) {
            int val = gen.randint(1, 10);
            REQUIRE(val >= 1);
            REQUIRE(val <= 10);
        }
    }
}

TEST_CASE("Matrix generation", "[utils]") {
    SECTION("Random uniform matrix") {
        Matrix mat = random_uniform(3, 4, -2.0, 2.0);
        
        REQUIRE(mat.rows() == 3);
        REQUIRE(mat.cols() == 4);
        
        // Check all values are in range
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                REQUIRE(mat(i, j) >= -2.0);
                REQUIRE(mat(i, j) < 2.0);
            }
        }
    }
    
    SECTION("Random normal matrix") {
        Matrix mat = random_normal(2, 3, 5.0, 2.0);
        
        REQUIRE(mat.rows() == 2);
        REQUIRE(mat.cols() == 3);
        
        // Just check that values are reasonable (not exact due to randomness)
        // Check that at least some values are not exactly zero (very low probability)
        bool has_nonzero = false;
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                if (std::abs(mat(i, j)) > 0.1) {
                    has_nonzero = true;
                    break;
                }
            }
        }
        REQUIRE(has_nonzero);
    }
    
    SECTION("Set seed affects matrix generation") {
        set_seed(123);
        Matrix mat1 = random_uniform(2, 2);
        
        set_seed(123);
        Matrix mat2 = random_uniform(2, 2);
        
        for (int i = 0; i < mat1.rows(); ++i) {
            for (int j = 0; j < mat1.cols(); ++j) {
                REQUIRE(mat1(i, j) == Catch::Approx(mat2(i, j)));
            }
        }
    }
}

TEST_CASE("Validation utilities", "[utils]") {
    SECTION("Check dimensions") {
        Matrix mat(3, 4);
        
        REQUIRE_NOTHROW(validation::check_dimensions(mat, 3, 4));
        REQUIRE_NOTHROW(validation::check_dimensions(mat, 3, -1)); // ignore cols
        REQUIRE_NOTHROW(validation::check_dimensions(mat, -1, 4)); // ignore rows
        REQUIRE_NOTHROW(validation::check_dimensions(mat, -1, -1)); // ignore both
        
        REQUIRE_THROWS_AS(validation::check_dimensions(mat, 2, 4), std::invalid_argument);
        REQUIRE_THROWS_AS(validation::check_dimensions(mat, 3, 3), std::invalid_argument);
    }
    
    SECTION("Check not empty") {
        Matrix mat(3, 4);
        Matrix empty_mat(0, 0);
        Matrix zero_rows(0, 4);
        Matrix zero_cols(3, 0);
        
        REQUIRE_NOTHROW(validation::check_not_empty(mat));
        REQUIRE_THROWS_AS(validation::check_not_empty(empty_mat), std::invalid_argument);
        REQUIRE_THROWS_AS(validation::check_not_empty(zero_rows), std::invalid_argument);
        REQUIRE_THROWS_AS(validation::check_not_empty(zero_cols), std::invalid_argument);
    }
    
    SECTION("Check vector size") {
        Vector vec(5);
        
        REQUIRE_NOTHROW(validation::check_vector_size(vec, 5));
        REQUIRE_NOTHROW(validation::check_vector_size(vec, -1)); // ignore size
        REQUIRE_THROWS_AS(validation::check_vector_size(vec, 4), std::invalid_argument);
        REQUIRE_THROWS_AS(validation::check_vector_size(vec, 6), std::invalid_argument);
    }
    
    SECTION("Check multiplication compatibility") {
        Matrix mat1(3, 4);
        Matrix mat2(4, 5);
        Matrix mat3(3, 5);
        
        REQUIRE_NOTHROW(validation::check_multiplication_compatible(mat1, mat2));
        REQUIRE_THROWS_AS(validation::check_multiplication_compatible(mat1, mat3), std::invalid_argument);
        REQUIRE_THROWS_AS(validation::check_multiplication_compatible(mat2, mat1), std::invalid_argument);
    }
}

TEST_CASE("Array utilities", "[utils]") {
    SECTION("Shape to string") {
        Shape shape1 = {3, 4};
        Shape shape2 = {10};
        Shape shape3 = {2, 3, 4};
        
        REQUIRE(array::shape_to_string(shape1) == "(3, 4)");
        REQUIRE(array::shape_to_string(shape2) == "(10)");
        REQUIRE(array::shape_to_string(shape3) == "(2, 3, 4)");
    }
    
    SECTION("Get shape") {
        Matrix mat(3, 4);
        Shape shape = array::get_shape(mat);
        
        REQUIRE(shape.size() == 2);
        REQUIRE(shape[0] == 3);
        REQUIRE(shape[1] == 4);
    }
    
    SECTION("Shapes equal") {
        Shape shape1 = {3, 4};
        Shape shape2 = {3, 4};
        Shape shape3 = {3, 5};
        Shape shape4 = {3, 4, 2};
        
        REQUIRE(array::shapes_equal(shape1, shape2));
        REQUIRE_FALSE(array::shapes_equal(shape1, shape3));
        REQUIRE_FALSE(array::shapes_equal(shape1, shape4));
    }
}