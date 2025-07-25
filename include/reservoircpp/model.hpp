/**
 * @file model.hpp
 * @brief Model class for ReservoirCpp - computational graph management
 * 
 * This file contains the C++ port of the Python model.py,
 * implementing the Model class for creating and managing
 * computational graphs of connected nodes.
 */

#ifndef RESERVOIRCPP_MODEL_HPP
#define RESERVOIRCPP_MODEL_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>
#include <functional>

#include "reservoircpp/types.hpp"
#include "reservoircpp/node.hpp"
#include "reservoircpp/utils.hpp"

namespace reservoircpp {

/**
 * @brief Edge representation in the computational graph
 */
using Edge = std::pair<NodePtr, NodePtr>;

/**
 * @brief Data dispatcher for managing input/output data flow in models
 */
class DataDispatcher {
public:
    /**
     * @brief Constructor
     * 
     * @param model Pointer to the model this dispatcher serves
     */
    explicit DataDispatcher(class Model* model) : model_(model) {}
    
    /**
     * @brief Load and distribute data to appropriate nodes
     * 
     * @param input Input data (can be single matrix or mapping)
     * @param target Target data (optional)
     * @return Data mapping for each node
     */
    std::unordered_map<std::string, Matrix> load(const Matrix& input, 
                                                 const Matrix* target = nullptr);
    
    /**
     * @brief Load data from mapping of node names to matrices
     * 
     * @param input_map Input data mapping
     * @param target_map Target data mapping (optional)
     * @return Data mapping for each node
     */
    std::unordered_map<std::string, Matrix> load(
        const std::unordered_map<std::string, Matrix>& input_map,
        const std::unordered_map<std::string, Matrix>* target_map = nullptr);

private:
    class Model* model_;
};

/**
 * @brief Model class for managing computational graphs of nodes
 * 
 * This is the main implementation of the Model concept from Python ReservoirPy.
 * It allows connecting nodes into computational graphs and manages data flow
 * between them.
 */
class Model : public Node {
public:
    /**
     * @brief Constructor for Model
     * 
     * @param nodes List of nodes to include in the model
     * @param edges List of edges connecting the nodes
     * @param name Model name (optional)
     */
    explicit Model(const std::vector<NodePtr>& nodes = {},
                   const std::vector<Edge>& edges = {},
                   const std::string& name = "");
    
    /**
     * @brief Virtual destructor
     */
    virtual ~Model() = default;
    
    // Core interface methods
    
    /**
     * @brief Forward pass through the model
     * 
     * @param input Input matrix
     * @return Output matrix from terminal nodes
     */
    Matrix forward(const Matrix& input) override;
    
    /**
     * @brief Initialize the model with input/output data
     * 
     * @param x Input data
     * @param y Output data
     */
    void do_initialize(const Matrix* x, const Matrix* y) override;
    
    /**
     * @brief Fit the model using input/target data
     * 
     * @param X Input data
     * @param y Target data
     */
    virtual void fit(const Matrix& X, const Matrix& y);
    
    /**
     * @brief Run the model on input data (forward pass only)
     * 
     * @param X Input data
     * @return Output data
     */
    virtual Matrix run(const Matrix& X);
    
    // Graph management methods
    
    /**
     * @brief Add a node to the model
     * 
     * @param node Node to add
     */
    void add_node(NodePtr node);
    
    /**
     * @brief Add an edge between two nodes
     * 
     * @param parent Parent node
     * @param child Child node
     */
    void add_edge(NodePtr parent, NodePtr child);
    
    /**
     * @brief Get a node by name
     * 
     * @param name Node name
     * @return Pointer to the node
     * @throws std::invalid_argument if node not found
     */
    NodePtr get_node(const std::string& name) const;
    
    /**
     * @brief Check if model contains a node
     * 
     * @param name Node name
     * @return true if node exists
     */
    bool has_node(const std::string& name) const;
    
    /**
     * @brief Get all nodes in the model
     * 
     * @return Vector of node pointers
     */
    const std::vector<NodePtr>& get_nodes() const { return nodes_; }
    
    /**
     * @brief Get all edges in the model
     * 
     * @return Vector of edges
     */
    const std::vector<Edge>& get_edges() const { return edges_; }
    
    /**
     * @brief Get input nodes (nodes with no parents)
     * 
     * @return Vector of input node pointers
     */
    const std::vector<NodePtr>& get_input_nodes() const { return input_nodes_; }
    
    /**
     * @brief Get output nodes (nodes with no children)
     * 
     * @return Vector of output node pointers
     */
    const std::vector<NodePtr>& get_output_nodes() const { return output_nodes_; }
    
    /**
     * @brief Get trainable nodes
     * 
     * @return Vector of trainable node pointers
     */
    std::vector<NodePtr> get_trainable_nodes() const;
    
    /**
     * @brief Check if model is empty
     * 
     * @return true if no nodes
     */
    bool is_empty() const { return nodes_.empty(); }
    
    /**
     * @brief Get node names
     * 
     * @return Vector of node names
     */
    std::vector<std::string> get_node_names() const;
    
    /**
     * @brief Update the graph structure (recalculate topology, inputs, outputs)
     */
    void update_graph();
    
    /**
     * @brief Reset all nodes in the model
     * 
     * @param state Optional state to reset to (unused in model context)
     */
    void reset(const Vector* state = nullptr) override;
    
    /**
     * @brief Copy the model
     * 
     * @param new_name New name for the copy
     * @return Shared pointer to the copied model
     */
    std::shared_ptr<Node> copy(const std::string& new_name = "") const override;

    // Property accessors
    
    /**
     * @brief Check if all trainable nodes are fitted
     * 
     * @return true if all trainable nodes are fitted
     */
    bool is_fitted() const;
    
    /**
     * @brief Get data dispatcher
     * 
     * @return Reference to data dispatcher
     */
    DataDispatcher& get_data_dispatcher() { return data_dispatcher_; }

protected:
    /**
     * @brief Find input and output nodes based on current edges
     * 
     * @param input_nodes Output parameter for input nodes
     * @param output_nodes Output parameter for output nodes
     */
    void find_input_output_nodes(std::vector<NodePtr>& input_nodes,
                                 std::vector<NodePtr>& output_nodes) const;
    
    /**
     * @brief Perform topological sort on nodes
     * 
     * @return Topologically sorted vector of nodes
     */
    std::vector<NodePtr> topological_sort() const;
    
    /**
     * @brief Check for cycles in the graph
     * 
     * @return true if graph has cycles
     */
    bool has_cycles() const;

private:
    std::vector<NodePtr> nodes_;
    std::vector<Edge> edges_;
    std::vector<NodePtr> input_nodes_;
    std::vector<NodePtr> output_nodes_;
    std::unordered_map<std::string, NodePtr> node_registry_;
    DataDispatcher data_dispatcher_;
    
    /**
     * @brief Generate unique model name
     * 
     * @param base_name Base name (empty for auto-generation)
     * @return Unique model name
     */
    std::string generate_model_name(const std::string& base_name = "") const;
};

/**
 * @brief Node connection operator - creates a model with connected nodes
 * 
 * @param left Left node
 * @param right Right node
 * @return Model with connected nodes
 */
std::shared_ptr<Model> operator>>(NodePtr left, NodePtr right);

/**
 * @brief Model connection operator - connects a node to a model
 * 
 * @param left Left node
 * @param right Right model
 * @return Updated model with connected node
 */
std::shared_ptr<Model> operator>>(NodePtr left, std::shared_ptr<Model> right);

/**
 * @brief Model connection operator - connects a model to a node
 * 
 * @param left Left model
 * @param right Right node
 * @return Updated model with connected node
 */
std::shared_ptr<Model> operator>>(std::shared_ptr<Model> left, NodePtr right);

/**
 * @brief Model merge operator - merges two models
 * 
 * @param left Left model
 * @param right Right model
 * @return Merged model
 */
std::shared_ptr<Model> operator&(std::shared_ptr<Model> left, std::shared_ptr<Model> right);

} // namespace reservoircpp

#endif // RESERVOIRCPP_MODEL_HPP