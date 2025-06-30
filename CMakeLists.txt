cmake_minimum_required(VERSION 3.12)
project(reservoircpp VERSION 0.1.0 LANGUAGES CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler flags for better debugging and optimization
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options for building components
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_DOCS "Build documentation" OFF)

# Find required packages
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(include)

# Create the main library
add_library(reservoircpp INTERFACE)
target_include_directories(reservoircpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(reservoircpp INTERFACE Eigen3::Eigen)
target_compile_features(reservoircpp INTERFACE cxx_std_17)

# Add subdirectories
add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS reservoircpp
    EXPORT reservoircppTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets
install(EXPORT reservoircppTargets
    FILE reservoircppTargets.cmake
    NAMESPACE reservoircpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/reservoircpp
)

# Create and install config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    reservoircppConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    cmake/reservoircppConfig.cmake.in
    reservoircppConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/reservoircpp
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/reservoircppConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/reservoircppConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/reservoircpp
)