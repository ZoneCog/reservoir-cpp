/**
 * @file reservoircpp.hpp
 * @brief Main header file for ReservoirCpp library
 * 
 * ReservoirCpp is a C++ implementation of reservoir computing algorithms,
 * ported from the Python ReservoirPy library.
 * 
 * @author ReservoirCpp Development Team
 * @date 2024
 * @license MIT License
 * 
 * This library provides efficient implementations of Echo State Networks (ESN)
 * and other reservoir computing architectures using modern C++17 features
 * and Eigen for linear algebra operations.
 */

#ifndef RESERVOIRCPP_HPP
#define RESERVOIRCPP_HPP

// Include version information
#include "reservoircpp/version.hpp"

// Core type definitions
#include "reservoircpp/types.hpp"

// Utility functions
#include "reservoircpp/utils.hpp"

// Activation functions
#include "reservoircpp/activations.hpp"

// Node base class
#include "reservoircpp/node.hpp"

// Will be expanded with more includes as components are implemented
// #include "reservoircpp/reservoir.hpp"
// #include "reservoircpp/readout.hpp"
// #include "reservoircpp/matrix_generators.hpp"

#endif // RESERVOIRCPP_HPP