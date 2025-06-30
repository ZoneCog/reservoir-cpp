/**
 * @file test_version.cpp
 * @brief Tests for version information
 */

#include <catch2/catch_test_macros.hpp>
#include <reservoircpp/version.hpp>

TEST_CASE("Version information", "[version]") {
    SECTION("Version string format") {
        REQUIRE(reservoircpp::version() == "0.1.0");
        REQUIRE(reservoircpp::VERSION_MAJOR == 0);
        REQUIRE(reservoircpp::VERSION_MINOR == 1);
        REQUIRE(reservoircpp::VERSION_PATCH == 0);
    }
    
    SECTION("Version info contains library name") {
        auto info = reservoircpp::version_info();
        REQUIRE(info.find("ReservoirCpp") != std::string::npos);
        REQUIRE(info.find("0.1.0") != std::string::npos);
    }
}