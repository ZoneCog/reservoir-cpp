/**
 * @file compat.hpp
 * @brief Compatibility utilities for ReservoirCpp
 * 
 * This module provides compatibility functions for loading and converting
 * models from different versions and formats, including Python ReservoirPy models.
 * 
 * @author ReservoirCpp Development Team
 * @date 2024
 * @license MIT License
 */

#ifndef RESERVOIRCPP_COMPAT_HPP
#define RESERVOIRCPP_COMPAT_HPP

#include "reservoircpp/types.hpp"
#include "reservoircpp/node.hpp"
#include "reservoircpp/reservoir.hpp"
#include "reservoircpp/readout.hpp"
#include <string>
#include <memory>
#include <unordered_map>

namespace reservoircpp {
namespace compat {

/**
 * @brief Model configuration structure for compatibility
 */
struct ModelConfig {
    std::string version;                    ///< Model version
    std::string model_type;                 ///< Type of model (ESN, etc.)
    std::unordered_map<std::string, float> parameters;  ///< Model parameters
    std::unordered_map<std::string, Matrix> matrices;   ///< Model matrices
    
    ModelConfig() = default;
};

/**
 * @brief Serialization utilities for model save/load
 */
class ModelSerializer {
public:
    /**
     * @brief Save a node to binary format
     * @param node Node to save
     * @param filename Output filename
     * @return True if successful
     */
    static bool save_node(const Node& node, const std::string& filename);

    /**
     * @brief Load a node from binary format
     * @param filename Input filename
     * @return Loaded node or nullptr if failed
     */
    static std::unique_ptr<Node> load_node(const std::string& filename);

    /**
     * @brief Save model configuration to JSON-like format
     * @param config Model configuration
     * @param filename Output filename
     * @return True if successful
     */
    static bool save_config(const ModelConfig& config, const std::string& filename);

    /**
     * @brief Load model configuration from file
     * @param filename Input filename
     * @return Loaded configuration
     */
    static ModelConfig load_config(const std::string& filename);

    /**
     * @brief Export model to Python-compatible format
     * @param node Node to export
     * @param directory Output directory
     * @return True if successful
     */
    static bool export_to_python(const Node& node, const std::string& directory);
};

/**
 * @brief Legacy model loader for ReservoirPy v0.2 compatibility
 */
class LegacyLoader {
public:
    /**
     * @brief Load ReservoirPy v0.2 ESN model
     * @param directory Directory containing model files
     * @return Loaded ESN model or nullptr if failed
     */
    static std::unique_ptr<Node> load_reservoirpy_v2(const std::string& directory);

    /**
     * @brief Convert Python numpy array file to Eigen matrix
     * @param filename .npy file path
     * @return Loaded matrix
     */
    static Matrix load_numpy_array(const std::string& filename);

    /**
     * @brief Parse JSON configuration file
     * @param filename JSON file path
     * @return Parsed configuration
     */
    static ModelConfig parse_json_config(const std::string& filename);

private:
    /**
     * @brief Helper to read binary data from file
     */
    static std::vector<uint8_t> read_binary_file(const std::string& filename);
    
    /**
     * @brief Helper to parse numpy array header
     */
    static std::tuple<std::string, std::vector<size_t>, bool> parse_numpy_header(
        const std::vector<uint8_t>& data, size_t& header_end);
};

/**
 * @brief Model conversion utilities
 */
class ModelConverter {
public:
    /**
     * @brief Convert legacy ESN to modern format
     * @param config Legacy configuration
     * @return Modern ESN model
     */
    static std::unique_ptr<Node> convert_legacy_esn(const ModelConfig& config);

    /**
     * @brief Update model to current version
     * @param node Model to update
     * @param target_version Target version string
     * @return Updated model
     */
    static std::unique_ptr<Node> update_model_version(
        const Node& node, const std::string& target_version);

    /**
     * @brief Validate model compatibility
     * @param config Model configuration
     * @return True if compatible
     */
    static bool validate_compatibility(const ModelConfig& config);

private:
    /**
     * @brief Apply version-specific migrations
     */
    static void apply_version_migrations(ModelConfig& config, 
                                       const std::string& from_version,
                                       const std::string& to_version);
};

/**
 * @brief Format detection utilities
 */
class FormatDetector {
public:
    /**
     * @brief Detect model format from directory
     * @param path Path to model files
     * @return Detected format string
     */
    static std::string detect_format(const std::string& path);

    /**
     * @brief Check if directory contains ReservoirPy v0.2 model
     * @param directory Directory path
     * @return True if v0.2 model detected
     */
    static bool is_reservoirpy_v2(const std::string& directory);

    /**
     * @brief Check if file is numpy array
     * @param filename File path
     * @return True if numpy array
     */
    static bool is_numpy_file(const std::string& filename);

    /**
     * @brief Check if file is JSON configuration
     * @param filename File path
     * @return True if JSON file
     */
    static bool is_json_config(const std::string& filename);
};

/**
 * @brief Version compatibility information
 */
struct VersionInfo {
    static constexpr const char* CURRENT_VERSION = "0.4.0";
    static constexpr const char* MIN_COMPATIBLE_VERSION = "0.2.0";
    
    /**
     * @brief Check if version is supported
     * @param version Version string to check
     * @return True if supported
     */
    static bool is_supported(const std::string& version);
    
    /**
     * @brief Compare version strings
     * @param v1 First version
     * @param v2 Second version  
     * @return -1 if v1 < v2, 0 if equal, 1 if v1 > v2
     */
    static int compare_versions(const std::string& v1, const std::string& v2);
};

} // namespace compat
} // namespace reservoircpp

#endif // RESERVOIRCPP_COMPAT_HPP