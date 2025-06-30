/**
 * @file test_types.cpp
 * @brief Tests for core type definitions
 */

#include <catch2/catch_test_macros.hpp>
#include <reservoircpp/types.hpp>

TEST_CASE("Basic type definitions", "[types]") {
    SECTION("Float type is double") {
        REQUIRE(std::is_same_v<reservoircpp::Float, double>);
    }
    
    SECTION("Matrix types are correctly defined") {
        reservoircpp::Matrix m(2, 2);
        m << 1.0, 2.0, 3.0, 4.0;
        
        REQUIRE(m.rows() == 2);
        REQUIRE(m.cols() == 2);
        REQUIRE(m(0, 0) == 1.0);
        REQUIRE(m(1, 1) == 4.0);
    }
    
    SECTION("Vector types work") {
        reservoircpp::Vector v(3);
        v << 1.0, 2.0, 3.0;
        
        REQUIRE(v.size() == 3);
        REQUIRE(v(0) == 1.0);
        REQUIRE(v(2) == 3.0);
    }
    
    SECTION("Shape type") {
        reservoircpp::Shape shape = {10, 20};
        REQUIRE(shape.size() == 2);
        REQUIRE(shape[0] == 10);
        REQUIRE(shape[1] == 20);
    }
}