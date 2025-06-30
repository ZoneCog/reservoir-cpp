/**
 * @file version.hpp
 * @brief Version information for ReservoirCpp
 */

#ifndef RESERVOIRCPP_VERSION_HPP
#define RESERVOIRCPP_VERSION_HPP

#include <string>

namespace reservoircpp {

constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 1;
constexpr int VERSION_PATCH = 0;

constexpr const char* VERSION_STRING = "0.1.0";

/**
 * @brief Get the version string
 * @return Version string in format "major.minor.patch"
 */
inline std::string version() {
    return VERSION_STRING;
}

/**
 * @brief Get detailed version information
 * @return Detailed version information including library name
 */
inline std::string version_info() {
    return "ReservoirCpp v" + version() + " - C++ Reservoir Computing Library";
}

} // namespace reservoircpp

#endif // RESERVOIRCPP_VERSION_HPP