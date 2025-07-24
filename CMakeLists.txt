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
option(BUILD_HEADER_ONLY "Build as header-only library" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Find required packages
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(include)

# Create the main library (now depends on core components)
if(BUILD_HEADER_ONLY)
    # Header-only library mode
    add_library(reservoircpp INTERFACE)
    target_include_directories(reservoircpp INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(reservoircpp INTERFACE Eigen3::Eigen)
    target_compile_features(reservoircpp INTERFACE cxx_std_17)
    target_compile_definitions(reservoircpp INTERFACE RESERVOIRCPP_HEADER_ONLY)
    
    # For header-only mode, we don't build the core library
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
else()
    # Normal library mode with compiled components
    add_library(reservoircpp INTERFACE)
    target_include_directories(reservoircpp INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(reservoircpp INTERFACE Eigen3::Eigen)
    target_compile_features(reservoircpp INTERFACE cxx_std_17)
    
    # Add subdirectories
    add_subdirectory(src)
    
    # Link the interface library with the core library
    target_link_libraries(reservoircpp INTERFACE reservoircpp_core)
endif()

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

# Install targets (conditional for header-only mode)
if(BUILD_HEADER_ONLY)
    install(TARGETS reservoircpp
        EXPORT reservoircppTargets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
else()
    install(TARGETS reservoircpp reservoircpp_core
        EXPORT reservoircppTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

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

# Generate and install pkgconfig file
if(NOT BUILD_HEADER_ONLY)
    configure_file(
        cmake/reservoircpp.pc.in
        reservoircpp.pc
        @ONLY
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/reservoircpp.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    )
endif()