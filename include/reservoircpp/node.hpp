/**
 * @file node.hpp
 * @brief Base Node class for ReservoirCpp
 * 
 * This file contains the C++ port of the Python node.py,
 * implementing the core Node functionality.
 */

#ifndef RESERVOIRCPP_NODE_HPP
#define RESERVOIRCPP_NODE_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <any>
#include <functional>
#include <stdexcept>
#include <sstream>
#include <random>
#include <chrono>
#include <typeindex>

#include "reservoircpp/types.hpp"
#include "reservoircpp/utils.hpp"

namespace reservoircpp {

/**
 * @brief Generate a unique identifier string
 * 
 * @return Unique identifier string
 */
inline std::string generate_uuid() {
    static std::mt19937 gen(std::chrono::steady_clock::now().time_since_epoch().count());
    static std::uniform_int_distribution<int> dis(0, 15);
    
    std::stringstream ss;
    ss << "node_";
    
    // Generate a simple hex identifier
    for (int i = 0; i < 8; ++i) {
        ss << std::hex << dis(gen);
    }
    
    return ss.str();
}

/**
 * @brief Base class for all nodes in the computational graph
 * 
 * This is the main implementation of the Node concept from Python ReservoirPy.
 * It provides the core functionality for stateful computation nodes.
 */
class Node : public NodeInterface {
public:
    /**
     * @brief Constructor for Node
     * 
     * @param name Node name (optional, will generate unique name if empty)
     * @param params Initial parameters
     * @param hypers Initial hyperparameters
     */
    explicit Node(const std::string& name = "", 
                  const ParameterMap& params = {},
                  const ParameterMap& hypers = {})
        : name_(name.empty() ? generate_uuid() : name)
        , params_(params)
        , hypers_(hypers)
        , is_initialized_(false)
        , is_trainable_(true)
        , input_dim_({})
        , output_dim_({})
        , state_()
        , dtype_(std::type_index(typeid(Float))) {
    }
    
    /**
     * @brief Virtual destructor
     */
    virtual ~Node() = default;
    
    // Core interface methods from NodeInterface
    
    /**
     * @brief Forward pass through the node
     * 
     * @param input Input matrix
     * @return Output matrix
     */
    Matrix operator()(const Matrix& input) override {
        if (!is_initialized_) {
            initialize(&input);
        }
        return forward(input);
    }
    
    /**
     * @brief Initialize the node with input/output data
     * 
     * @param x Input data (can be null)
     * @param y Output data (can be null)
     */
    void initialize(const Matrix* x = nullptr, const Matrix* y = nullptr) override {
        if (is_initialized_) {
            return; // Already initialized
        }
        
        if (x != nullptr) {
            set_input_dim({static_cast<int>(x->rows()), static_cast<int>(x->cols())});
        }
        
        if (y != nullptr) {
            set_output_dim({static_cast<int>(y->rows()), static_cast<int>(y->cols())});
        }
        
        // Call virtual initialization method
        do_initialize(x, y);
        
        is_initialized_ = true;
        
        // Initialize state after marking as initialized
        reset();
    }
    
    /**
     * @brief Reset node state
     * 
     * @param state New state (null for zero state)
     */
    void reset(const Vector* state = nullptr) override {
        if (state != nullptr) {
            if (is_initialized_) {
                utils::validation::check_vector_size(*state, get_output_size(), "reset state");
            }
            state_ = *state;
        } else {
            state_ = zero_state();
        }
    }
    
    // Parameter management
    
    /**
     * @brief Get parameter value
     * 
     * @param name Parameter name
     * @return Parameter value
     * @throws std::invalid_argument if parameter not found
     */
    std::any get_param(const std::string& name) const override {
        auto it = params_.find(name);
        if (it != params_.end()) {
            return it->second;
        }
        
        auto hyper_it = hypers_.find(name);
        if (hyper_it != hypers_.end()) {
            return hyper_it->second;
        }
        
        throw std::invalid_argument("No parameter named '" + name + "' found in node " + name_);
    }
    
    /**
     * @brief Set parameter value
     * 
     * @param name Parameter name
     * @param value Parameter value
     * @throws std::invalid_argument if parameter not found
     */
    void set_param(const std::string& name, const std::any& value) override {
        auto it = params_.find(name);
        if (it != params_.end()) {
            params_[name] = value;
            return;
        }
        
        auto hyper_it = hypers_.find(name);
        if (hyper_it != hypers_.end()) {
            hypers_[name] = value;
            return;
        }
        
        throw std::invalid_argument("No parameter named '" + name + "' found in node " + name_);
    }
    
    // Property accessors
    
    std::string name() const override { return name_; }
    void set_name(const std::string& name) override { name_ = name; }
    bool is_initialized() const override { return is_initialized_; }
    bool is_trainable() const override { return is_trainable_; }
    Shape input_dim() const override { return input_dim_; }
    Shape output_dim() const override { return output_dim_; }
    
    // State management
    
    Vector get_state() const override { 
        return state_; 
    }
    void set_state(const Vector& state) override { 
        utils::validation::check_vector_size(state, get_output_size(), "node state");
        state_ = state; 
    }
    
    /**
     * @brief Get zero state vector
     * 
     * @return Zero state vector
     */
    Vector zero_state() const {
        int size = get_output_size();
        if (size == 0) {
            // If no output size is set, return empty vector
            return Vector(0);
        }
        return Vector::Zero(size);
    }
    
    /**
     * @brief Set input dimension
     * 
     * @param dim Input dimension
     */
    void set_input_dim(const Shape& dim) {
        if (is_initialized_) {
            throw std::runtime_error("Cannot change input dimension after initialization");
        }
        input_dim_ = dim;
    }
    
    /**
     * @brief Set output dimension
     * 
     * @param dim Output dimension
     */
    void set_output_dim(const Shape& dim) {
        if (is_initialized_) {
            throw std::runtime_error("Cannot change output dimension after initialization");
        }
        output_dim_ = dim;
    }
    
    /**
     * @brief Get input size (product of input dimensions)
     * 
     * @return Input size
     */
    int get_input_size() const {
        if (input_dim_.empty()) return 0;
        int size = 1;
        for (int dim : input_dim_) {
            size *= dim;
        }
        return size;
    }
    
    /**
     * @brief Get output size (product of output dimensions)
     * 
     * @return Output size
     */
    int get_output_size() const {
        if (output_dim_.empty()) return 0;
        int size = 1;
        for (int dim : output_dim_) {
            size *= dim;
        }
        return size;
    }
    
    /**
     * @brief Check if node has parameter
     * 
     * @param name Parameter name
     * @return true if parameter exists
     */
    bool has_param(const std::string& name) const {
        return params_.find(name) != params_.end() || 
               hypers_.find(name) != hypers_.end();
    }
    
    /**
     * @brief Get all parameter names
     * 
     * @return Vector of parameter names
     */
    std::vector<std::string> get_param_names() const {
        std::vector<std::string> names;
        for (const auto& pair : params_) {
            names.push_back(pair.first);
        }
        for (const auto& pair : hypers_) {
            names.push_back(pair.first);
        }
        return names;
    }
    
    /**
     * @brief Copy the node
     * 
     * @param new_name New name for the copy (empty to generate unique name)
     * @return Shared pointer to the copied node
     */
    virtual std::shared_ptr<Node> copy(const std::string& new_name = "") const {
        std::string copy_name = new_name.empty() ? generate_uuid() : new_name;
        auto node_copy = std::make_shared<Node>(copy_name, params_, hypers_);
        
        // Copy state
        node_copy->input_dim_ = input_dim_;
        node_copy->output_dim_ = output_dim_;
        node_copy->state_ = state_;
        node_copy->is_initialized_ = is_initialized_;
        node_copy->is_trainable_ = is_trainable_;
        
        return node_copy;
    }
    
protected:
    /**
     * @brief Virtual forward pass implementation
     * 
     * Subclasses should override this method to implement their specific forward pass.
     * 
     * @param input Input matrix
     * @return Output matrix
     */
    virtual Matrix forward(const Matrix& input) {
        // Default implementation: just return input (identity)
        return input;
    }
    
    /**
     * @brief Virtual initialization implementation
     * 
     * Subclasses should override this method to implement their specific initialization.
     * 
     * @param x Input data
     * @param y Output data
     */
    virtual void do_initialize(const Matrix* x, const Matrix* y) {
        // Default implementation: do nothing
        (void)x; // Suppress unused parameter warning
        (void)y;
    }
    
    /**
     * @brief Get mutable reference to parameters
     * 
     * @return Reference to parameters map
     */
    ParameterMap& get_params() { return params_; }
    
    /**
     * @brief Get mutable reference to hyperparameters
     * 
     * @return Reference to hyperparameters map
     */
    ParameterMap& get_hypers() { return hypers_; }
    
    /**
     * @brief Get const reference to parameters
     * 
     * @return Const reference to parameters map
     */
    const ParameterMap& get_params() const { return params_; }
    
    /**
     * @brief Get const reference to hyperparameters
     * 
     * @return Const reference to hyperparameters map
     */
    const ParameterMap& get_hypers() const { return hypers_; }

private:
    std::string name_;
    ParameterMap params_;
    ParameterMap hypers_;
    bool is_initialized_;
    bool is_trainable_;
    Shape input_dim_;
    Shape output_dim_;
    Vector state_;
    std::type_index dtype_;
};

} // namespace reservoircpp

#endif // RESERVOIRCPP_NODE_HPP