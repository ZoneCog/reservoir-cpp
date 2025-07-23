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

// Matrix generation utilities
#include "reservoircpp/matrix_generators.hpp"

// Reservoir components
#include "reservoircpp/reservoir.hpp"

// Readout components
#include "reservoircpp/readout.hpp"

// Observables and metrics
#include "reservoircpp/observables.hpp"

// Datasets and data utilities
#include "reservoircpp/datasets.hpp"

// Stage 5: Ancillary and Advanced Features
#include "reservoircpp/compat.hpp"
#include "reservoircpp/experimental.hpp"
#include "reservoircpp/hyper.hpp"
#include "reservoircpp/plotting.hpp"
#include "reservoircpp/gpu.hpp"

#endif // RESERVOIRCPP_HPP