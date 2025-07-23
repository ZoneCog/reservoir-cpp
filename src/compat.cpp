/**
 * @file compat.cpp
 * @brief Implementation of compatibility utilities
 */

#include "reservoircpp/compat.hpp"
#include "reservoircpp/utils.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <regex>

namespace reservoircpp {
namespace compat {

// ModelSerializer Implementation
bool ModelSerializer::save_node(const Node& node, const std::string& filename) {
    try {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        
        // Save basic node information
        std::string name = node.name();
        size_t name_size = name.size();
        file.write(reinterpret_cast<const char*>(&name_size), sizeof(name_size));
        file.write(name.c_str(), name_size);
        
        // Save output dimensions
        auto output_dims = node.output_dim();
        size_t dims_size = output_dims.size();
        file.write(reinterpret_cast<const char*>(&dims_size), sizeof(dims_size));
        for (int dim : output_dims) {
            size_t dim_size = static_cast<size_t>(dim);
            file.write(reinterpret_cast<const char*>(&dim_size), sizeof(dim_size));
        }
        
        // Save parameters - accessing parameters directly via get_params
        auto params = node.get_params();
        size_t params_size = params.size();
        file.write(reinterpret_cast<const char*>(&params_size), sizeof(params_size));
        
        for (const auto& [key, value] : params) {
            size_t key_size = key.size();
            file.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
            file.write(key.c_str(), key_size);
            // For simplicity, we'll serialize all parameters as float
            try {
                float float_val = std::any_cast<float>(value);
                file.write(reinterpret_cast<const char*>(&float_val), sizeof(float_val));
            } catch (const std::bad_any_cast&) {
                float default_val = 0.0f;
                file.write(reinterpret_cast<const char*>(&default_val), sizeof(default_val));
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving node: " << e.what() << std::endl;
        return false;
    }
}

std::unique_ptr<Node> ModelSerializer::load_node(const std::string& filename) {
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return nullptr;
        }
        
        // Load name
        size_t name_size;
        file.read(reinterpret_cast<char*>(&name_size), sizeof(name_size));
        std::string name(name_size, '\0');
        file.read(&name[0], name_size);
        
        // Load output dimensions
        size_t dims_size;
        file.read(reinterpret_cast<char*>(&dims_size), sizeof(dims_size));
        std::vector<int> output_dims(dims_size);
        for (size_t i = 0; i < dims_size; ++i) {
            size_t dim_size;
            file.read(reinterpret_cast<char*>(&dim_size), sizeof(dim_size));
            output_dims[i] = static_cast<int>(dim_size);
        }
        
        // Load parameters
        size_t params_size;
        file.read(reinterpret_cast<char*>(&params_size), sizeof(params_size));
        
        auto node = std::make_unique<Node>(name);
        node->set_output_dim(output_dims);
        
        for (size_t i = 0; i < params_size; ++i) {
            size_t key_size;
            file.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
            std::string key(key_size, '\0');
            file.read(&key[0], key_size);
            
            float value;
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            node->get_params()[key] = value;
        }
        
        return node;
    } catch (const std::exception& e) {
        std::cerr << "Error loading node: " << e.what() << std::endl;
        return nullptr;
    }
}

bool ModelSerializer::save_config(const ModelConfig& config, const std::string& filename) {
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        // Simple JSON-like format
        file << "{\n";
        file << "  \"version\": \"" << config.version << "\",\n";
        file << "  \"model_type\": \"" << config.model_type << "\",\n";
        
        file << "  \"parameters\": {\n";
        bool first = true;
        for (const auto& [key, value] : config.parameters) {
            if (!first) file << ",\n";
            file << "    \"" << key << "\": " << value;
            first = false;
        }
        file << "\n  },\n";
        
        file << "  \"matrices\": {\n";
        first = true;
        for (const auto& [key, matrix] : config.matrices) {
            if (!first) file << ",\n";
            file << "    \"" << key << "\": [" << matrix.rows() << ", " << matrix.cols() << "]";
            first = false;
        }
        file << "\n  }\n";
        file << "}\n";
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving config: " << e.what() << std::endl;
        return false;
    }
}

ModelConfig ModelSerializer::load_config(const std::string& filename) {
    ModelConfig config;
    
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return config;
        }
        
        std::string line, content;
        while (std::getline(file, line)) {
            content += line;
        }
        
        // Simple parsing (would need proper JSON parser in real implementation)
        std::regex version_regex("\"version\":\\s*\"([^\"]+)\"");
        std::regex model_type_regex("\"model_type\":\\s*\"([^\"]+)\"");
        
        std::smatch match;
        if (std::regex_search(content, match, version_regex)) {
            config.version = match[1].str();
        }
        if (std::regex_search(content, match, model_type_regex)) {
            config.model_type = match[1].str();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
    }
    
    return config;
}

bool ModelSerializer::export_to_python(const Node& node, const std::string& directory) {
    try {
        // Create directory if it doesn't exist (manual implementation for compatibility)
        std::string mkdir_cmd = "mkdir -p " + directory;
        system(mkdir_cmd.c_str());
        
        // Export node configuration
        ModelConfig config;
        config.version = VersionInfo::CURRENT_VERSION;
        config.model_type = "Node";
        
        // Create simple model configuration
        ModelConfig config;
        config.version = VersionInfo::CURRENT_VERSION;
        config.model_type = "Node";
        
        // For simplicity, we'll just save the node name and type
        // In a real implementation, we'd need public accessors for parameters
        return save_config(config, directory + "/config.json");
        
        return save_config(config, directory + "/config.json");
        
    } catch (const std::exception& e) {
        std::cerr << "Error exporting to Python: " << e.what() << std::endl;
        return false;
    }
}

// LegacyLoader Implementation
std::unique_ptr<Node> LegacyLoader::load_reservoirpy_v2(const std::string& directory) {
    try {
        // Check if directory exists (simplified check)
        std::ifstream check(directory);
        if (!check.good()) {
            return nullptr;
        }
        
        // Look for configuration files (simplified - look for a default name)
        std::string config_file = directory + "/config.json";
        std::ifstream test_file(config_file);
        if (!test_file.good()) {
            return nullptr;
        }
        
        ModelConfig config = parse_json_config(config_file);
        return ModelConverter::convert_legacy_esn(config);
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading ReservoirPy v2 model: " << e.what() << std::endl;
        return nullptr;
    }
}

Matrix LegacyLoader::load_numpy_array(const std::string& filename) {
    // Simplified numpy loader - would need proper implementation
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return Matrix();
        }
        
        // For now, return empty matrix
        // Real implementation would parse .npy format
        return Matrix::Zero(1, 1);
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading numpy array: " << e.what() << std::endl;
        return Matrix();
    }
}

ModelConfig LegacyLoader::parse_json_config(const std::string& filename) {
    return ModelSerializer::load_config(filename);
}

std::vector<uint8_t> LegacyLoader::read_binary_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return {};
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);
    
    return data;
}

std::tuple<std::string, std::vector<size_t>, bool> LegacyLoader::parse_numpy_header(
    const std::vector<uint8_t>& data, size_t& header_end) {
    
    // Simplified implementation
    header_end = 0;
    return std::make_tuple("float32", std::vector<size_t>{1, 1}, false);
}

// ModelConverter Implementation
std::unique_ptr<Node> ModelConverter::convert_legacy_esn(const ModelConfig& config) {
    auto node = std::make_unique<Node>("ConvertedESN");
    
    // Set basic parameters from config
    for (const auto& [key, value] : config.parameters) {
        node->set_param(key, value);
    }
    
    return node;
}

std::unique_ptr<Node> ModelConverter::update_model_version(
    const Node& node, const std::string& target_version) {
    
    // Create a copy of the node with updated version
    auto updated_node = std::make_unique<Node>(node.name());
    updated_node->set_output_dim(node.output_dim());
    
    // For simplicity, create a basic copy without accessing protected parameters
    // In a real implementation, we'd need public parameter accessors
    
    return updated_node;
}

bool ModelConverter::validate_compatibility(const ModelConfig& config) {
    return VersionInfo::is_supported(config.version);
}

void ModelConverter::apply_version_migrations(ModelConfig& config, 
                                            const std::string& from_version,
                                            const std::string& to_version) {
    // Apply version-specific migrations
    if (VersionInfo::compare_versions(from_version, "0.3.0") < 0) {
        // Migrate from v0.2 to v0.3
        // Add any necessary parameter transformations
    }
}

// FormatDetector Implementation
std::string FormatDetector::detect_format(const std::string& path) {
    // Simplified detection - check if it's a directory with config.json
    std::string config_path = path + "/config.json";
    std::ifstream config_file(config_path);
    if (config_file.good()) {
        return "reservoirpy_v2";
    }
    
    return "unknown";
}

bool FormatDetector::is_reservoirpy_v2(const std::string& directory) {
    // Simplified check - look for config.json and at least one .npy file
    std::string config_path = directory + "/config.json";
    std::string npy_path = directory + "/W.npy";  // Common reservoir weight file
    
    std::ifstream config_file(config_path);
    std::ifstream npy_file(npy_path);
    
    return config_file.good() && npy_file.good();
}

bool FormatDetector::is_numpy_file(const std::string& filename) {
    return filename.substr(filename.find_last_of(".") + 1) == "npy";
}

bool FormatDetector::is_json_config(const std::string& filename) {
    return filename.substr(filename.find_last_of(".") + 1) == "json";
}

// VersionInfo Implementation
bool VersionInfo::is_supported(const std::string& version) {
    return compare_versions(version, MIN_COMPATIBLE_VERSION) >= 0;
}

int VersionInfo::compare_versions(const std::string& v1, const std::string& v2) {
    // Simple version comparison
    std::vector<int> ver1, ver2;
    
    auto parse_version = [](const std::string& v) -> std::vector<int> {
        std::vector<int> parts;
        std::stringstream ss(v);
        std::string part;
        
        while (std::getline(ss, part, '.')) {
            parts.push_back(std::stoi(part));
        }
        return parts;
    };
    
    ver1 = parse_version(v1);
    ver2 = parse_version(v2);
    
    for (size_t i = 0; i < std::max(ver1.size(), ver2.size()); ++i) {
        int p1 = i < ver1.size() ? ver1[i] : 0;
        int p2 = i < ver2.size() ? ver2[i] : 0;
        
        if (p1 < p2) return -1;
        if (p1 > p2) return 1;
    }
    
    return 0;
}

} // namespace compat
} // namespace reservoircpp