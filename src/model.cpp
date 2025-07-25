/**
 * @file model.cpp
 * @brief Implementation of Model class for ReservoirCpp
 */

#include "reservoircpp/model.hpp"
#include <queue>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <functional>

namespace reservoircpp {

// DataDispatcher implementation

std::unordered_map<std::string, Matrix> DataDispatcher::load(const Matrix& input, 
                                                            const Matrix* target) {
    std::unordered_map<std::string, Matrix> data_map;
    
    if (!model_) {
        throw std::runtime_error("DataDispatcher: model is null");
    }
    
    // For single input, distribute to all input nodes
    const auto& input_nodes = model_->get_input_nodes();
    
    if (input_nodes.empty()) {
        throw std::runtime_error("Model has no input nodes");
    }
    
    // If there's only one input node, use the input directly
    if (input_nodes.size() == 1) {
        data_map[input_nodes[0]->name()] = input;
    } else {
        // For multiple input nodes, we need to split the input or replicate it
        // For simplicity, replicate the same input to all input nodes
        for (const auto& node : input_nodes) {
            data_map[node->name()] = input;
        }
    }
    
    return data_map;
}

std::unordered_map<std::string, Matrix> DataDispatcher::load(
    const std::unordered_map<std::string, Matrix>& input_map,
    const std::unordered_map<std::string, Matrix>* target_map) {
    
    // For mapped input, return as-is (with validation)
    std::unordered_map<std::string, Matrix> data_map = input_map;
    
    if (!model_) {
        throw std::runtime_error("DataDispatcher: model is null");
    }
    
    // Validate that all referenced nodes exist
    for (const auto& pair : input_map) {
        if (!model_->has_node(pair.first)) {
            throw std::invalid_argument("Input references unknown node: " + pair.first);
        }
    }
    
    if (target_map) {
        for (const auto& pair : *target_map) {
            if (!model_->has_node(pair.first)) {
                throw std::invalid_argument("Target references unknown node: " + pair.first);
            }
            // For now, we don't handle target data explicitly
            // This could be extended for supervised learning scenarios
        }
    }
    
    return data_map;
}

// Model implementation

Model::Model(const std::vector<NodePtr>& nodes,
             const std::vector<Edge>& edges,
             const std::string& name)
    : Node(name.empty() ? generate_model_name() : name)
    , nodes_(nodes)
    , edges_(edges)
    , data_dispatcher_(this) {
    
    // Build node registry
    for (const auto& node : nodes_) {
        if (!node) {
            throw std::invalid_argument("Model: null node pointer provided");
        }
        
        if (node_registry_.find(node->name()) != node_registry_.end()) {
            throw std::invalid_argument("Model: duplicate node name: " + node->name());
        }
        
        node_registry_[node->name()] = node;
    }
    
    // Validate edges
    for (const auto& edge : edges_) {
        if (!edge.first || !edge.second) {
            throw std::invalid_argument("Model: null node in edge");
        }
        
        if (node_registry_.find(edge.first->name()) == node_registry_.end() ||
            node_registry_.find(edge.second->name()) == node_registry_.end()) {
            throw std::invalid_argument("Model: edge references unknown node");
        }
    }
    
    update_graph();
}

Matrix Model::forward(const Matrix& input) {
    if (!is_initialized()) {
        initialize(&input);
    }
    
    // Load input data through dispatcher
    auto data_map = data_dispatcher_.load(input);
    
    // Execute nodes in topological order
    for (const auto& node : nodes_) {
        // Check if this node has input data or gets data from parent nodes
        Matrix node_input;
        bool has_input = false;
        
        // First check if there's direct input for this node
        auto data_it = data_map.find(node->name());
        if (data_it != data_map.end()) {
            node_input = data_it->second;
            has_input = true;
        } else {
            // Look for parent nodes that provide input
            for (const auto& edge : edges_) {
                if (edge.second->name() == node->name()) {
                    // This node is a child - get input from parent
                    if (!has_input) {
                        node_input = edge.first->get_state();
                        has_input = true;
                    } else {
                        // Multiple parents - for now, just concatenate or sum
                        // This is a simplification - real implementation might be more complex
                        Matrix parent_output = edge.first->get_state();
                        if (node_input.rows() == parent_output.rows()) {
                            node_input += parent_output;  // Element-wise addition
                        }
                    }
                }
            }
        }
        
        if (has_input) {
            // Call the node's forward method
            Matrix output = (*node)(node_input);
            // Node state is automatically updated by the call operator
        }
    }
    
    // Collect outputs from terminal nodes
    if (output_nodes_.empty()) {
        throw std::runtime_error("Model has no output nodes");
    }
    
    if (output_nodes_.size() == 1) {
        Vector state = output_nodes_[0]->get_state();
        if (state.size() == 0) {
            // If node state is empty, return a minimal valid result
            return Matrix::Ones(1, 1);
        }
        // Convert Vector to Matrix for consistency
        Matrix result(state.size(), 1);
        result.col(0) = state;
        return result;
    } else {
        // Multiple outputs - concatenate them vertically
        int total_rows = 0;
        
        for (const auto& out_node : output_nodes_) {
            Vector state = out_node->get_state();
            total_rows += state.size();
        }
        
        if (total_rows == 0) {
            return Matrix(1, 1);  // Return minimal valid matrix
        }
        
        Matrix result(total_rows, 1);
        int current_row = 0;
        
        for (const auto& out_node : output_nodes_) {
            Vector state = out_node->get_state();
            if (state.size() > 0) {
                result.block(current_row, 0, state.size(), 1) = state;
                current_row += state.size();
            }
        }
        
        return result;
    }
}

void Model::do_initialize(const Matrix* x, const Matrix* y) {
    if (nodes_.empty()) {
        return;
    }
    
    // Initialize all nodes
    for (const auto& node : nodes_) {
        if (!node->is_initialized()) {
            // For input nodes, use provided data
            bool is_input = std::find(input_nodes_.begin(), input_nodes_.end(), node) != input_nodes_.end();
            if (is_input && x) {
                node->initialize(x, y);
            } else {
                node->initialize();
            }
        }
    }
}

void Model::fit(const Matrix& X, const Matrix& y) {
    if (!is_initialized()) {
        initialize(&X, &y);
    }
    
    // This is a simplified implementation
    // Real implementation would handle sequence data and proper training
    
    // For each time step (assuming X is sequence data)
    int seq_len = X.rows();
    
    for (int t = 0; t < seq_len; ++t) {
        Matrix x_t = X.row(t);
        Matrix y_t = y.row(t);
        
        // Forward pass
        forward(x_t);
        
        // Training pass for trainable nodes
        auto trainable_nodes = get_trainable_nodes();
        for (const auto& node : trainable_nodes) {
            // This would need to be implemented per node type
            // For now, just a placeholder
        }
    }
}

Matrix Model::run(const Matrix& X) {
    if (!is_initialized()) {
        initialize(&X);
    }
    
    // For sequence data, run forward pass for each timestep
    int seq_len = X.rows();
    if (seq_len == 1) {
        return forward(X);
    }
    
    // Handle sequence data
    std::vector<Matrix> outputs;
    for (int t = 0; t < seq_len; ++t) {
        Matrix x_t = X.row(t);
        Matrix output = forward(x_t);
        outputs.push_back(output);
    }
    
    // Concatenate outputs
    if (outputs.empty()) {
        return Matrix(0, 0);
    }
    
    Matrix result(outputs.size(), outputs[0].cols());
    for (size_t i = 0; i < outputs.size(); ++i) {
        result.row(i) = outputs[i];
    }
    
    return result;
}

void Model::add_node(NodePtr node) {
    if (!node) {
        throw std::invalid_argument("Cannot add null node to model");
    }
    
    if (has_node(node->name())) {
        throw std::invalid_argument("Node with name '" + node->name() + "' already exists in model");
    }
    
    nodes_.push_back(node);
    node_registry_[node->name()] = node;
    update_graph();
}

void Model::add_edge(NodePtr parent, NodePtr child) {
    if (!parent || !child) {
        throw std::invalid_argument("Cannot create edge with null nodes");
    }
    
    if (!has_node(parent->name()) || !has_node(child->name())) {
        throw std::invalid_argument("Cannot create edge: one or both nodes not in model");
    }
    
    // Check if edge already exists
    for (const auto& edge : edges_) {
        if (edge.first->name() == parent->name() && edge.second->name() == child->name()) {
            return;  // Edge already exists
        }
    }
    
    edges_.emplace_back(parent, child);
    update_graph();
}

NodePtr Model::get_node(const std::string& name) const {
    auto it = node_registry_.find(name);
    if (it == node_registry_.end()) {
        throw std::invalid_argument("Node '" + name + "' not found in model");
    }
    return it->second;
}

bool Model::has_node(const std::string& name) const {
    return node_registry_.find(name) != node_registry_.end();
}

std::vector<NodePtr> Model::get_trainable_nodes() const {
    std::vector<NodePtr> trainable;
    for (const auto& node : nodes_) {
        if (node->is_trainable()) {
            trainable.push_back(node);
        }
    }
    return trainable;
}

std::vector<std::string> Model::get_node_names() const {
    std::vector<std::string> names;
    for (const auto& node : nodes_) {
        names.push_back(node->name());
    }
    return names;
}

void Model::update_graph() {
    if (nodes_.empty()) {
        input_nodes_.clear();
        output_nodes_.clear();
        return;
    }
    
    find_input_output_nodes(input_nodes_, output_nodes_);
    
    // Check for cycles first
    if (has_cycles()) {
        throw std::runtime_error("Model contains cycles - invalid graph structure");
    }
    
    // Sort nodes topologically
    nodes_ = topological_sort();
}

void Model::reset(const Vector* state) {
    for (const auto& node : nodes_) {
        node->reset();
    }
}

std::shared_ptr<Node> Model::copy(const std::string& new_name) const {
    std::string copy_name = new_name.empty() ? generate_model_name() : new_name;
    
    // Copy all nodes
    std::vector<NodePtr> copied_nodes;
    std::unordered_map<std::string, NodePtr> name_mapping;
    
    for (const auto& node : nodes_) {
        auto copied_node = std::static_pointer_cast<Node>(node->copy(node->name() + "_copy"));
        copied_nodes.push_back(copied_node);
        name_mapping[node->name()] = copied_node;
    }
    
    // Copy edges with new node references
    std::vector<Edge> copied_edges;
    for (const auto& edge : edges_) {
        auto parent_it = name_mapping.find(edge.first->name());
        auto child_it = name_mapping.find(edge.second->name());
        
        if (parent_it != name_mapping.end() && child_it != name_mapping.end()) {
            copied_edges.emplace_back(parent_it->second, child_it->second);
        }
    }
    
    auto copied_model = std::make_shared<Model>(copied_nodes, copied_edges, copy_name);
    
    return copied_model;
}

bool Model::is_fitted() const {
    for (const auto& node : nodes_) {
        if (node->is_trainable() && !node->is_initialized()) {
            return false;
        }
    }
    return true;
}

void Model::find_input_output_nodes(std::vector<NodePtr>& input_nodes,
                                   std::vector<NodePtr>& output_nodes) const {
    input_nodes.clear();
    output_nodes.clear();
    
    std::unordered_set<std::string> has_parents;
    std::unordered_set<std::string> has_children;
    
    for (const auto& edge : edges_) {
        has_children.insert(edge.first->name());
        has_parents.insert(edge.second->name());
    }
    
    for (const auto& node : nodes_) {
        if (has_parents.find(node->name()) == has_parents.end()) {
            input_nodes.push_back(node);
        }
        if (has_children.find(node->name()) == has_children.end()) {
            output_nodes.push_back(node);
        }
    }
}

std::vector<NodePtr> Model::topological_sort() const {
    if (nodes_.empty()) {
        return {};
    }
    
    // Kahn's algorithm
    std::unordered_map<std::string, int> in_degree;
    std::unordered_map<std::string, std::vector<NodePtr>> adjacency;
    
    // Initialize in-degree and adjacency list
    for (const auto& node : nodes_) {
        in_degree[node->name()] = 0;
        adjacency[node->name()] = {};
    }
    
    for (const auto& edge : edges_) {
        in_degree[edge.second->name()]++;
        adjacency[edge.first->name()].push_back(edge.second);
    }
    
    // Queue for nodes with no incoming edges
    std::queue<NodePtr> queue;
    for (const auto& node : nodes_) {
        if (in_degree[node->name()] == 0) {
            queue.push(node);
        }
    }
    
    std::vector<NodePtr> sorted_nodes;
    
    while (!queue.empty()) {
        NodePtr current = queue.front();
        queue.pop();
        sorted_nodes.push_back(current);
        
        for (const auto& neighbor : adjacency[current->name()]) {
            in_degree[neighbor->name()]--;
            if (in_degree[neighbor->name()] == 0) {
                queue.push(neighbor);
            }
        }
    }
    
    return sorted_nodes;
}

bool Model::has_cycles() const {
    if (nodes_.empty()) {
        return false;
    }
    
    // Use a more robust cycle detection using DFS with colors
    // White (0) = unvisited, Gray (1) = visiting, Black (2) = visited
    std::unordered_map<std::string, int> colors;
    std::unordered_map<std::string, std::vector<NodePtr>> adjacency;
    
    // Initialize colors and adjacency list
    for (const auto& node : nodes_) {
        colors[node->name()] = 0;  // White
        adjacency[node->name()] = {};
    }
    
    for (const auto& edge : edges_) {
        adjacency[edge.first->name()].push_back(edge.second);
    }
    
    // DFS to detect cycles
    std::function<bool(const std::string&)> has_cycle_dfs = [&](const std::string& node_name) -> bool {
        colors[node_name] = 1;  // Gray (visiting)
        
        for (const auto& neighbor : adjacency[node_name]) {
            const std::string& neighbor_name = neighbor->name();
            
            if (colors[neighbor_name] == 1) {  // Gray neighbor = back edge = cycle
                return true;
            }
            
            if (colors[neighbor_name] == 0 && has_cycle_dfs(neighbor_name)) {
                return true;
            }
        }
        
        colors[node_name] = 2;  // Black (visited)
        return false;
    };
    
    // Check each unvisited node
    for (const auto& node : nodes_) {
        if (colors[node->name()] == 0) {
            if (has_cycle_dfs(node->name())) {
                return true;
            }
        }
    }
    
    return false;
}

std::string Model::generate_model_name(const std::string& base_name) const {
    if (!base_name.empty()) {
        return base_name;
    }
    
    std::stringstream ss;
    ss << "model_";
    
    auto now = std::chrono::steady_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    ss << timestamp;
    
    return ss.str();
}

// Connection operators

std::shared_ptr<Model> operator>>(NodePtr left, NodePtr right) {
    if (!left || !right) {
        throw std::invalid_argument("Cannot connect null nodes");
    }
    
    std::vector<NodePtr> nodes = {left, right};
    std::vector<Edge> edges = {{left, right}};
    
    return std::make_shared<Model>(nodes, edges);
}

std::shared_ptr<Model> operator>>(NodePtr left, std::shared_ptr<Model> right) {
    if (!left || !right) {
        throw std::invalid_argument("Cannot connect null node/model");
    }
    
    // Create new model that includes the left node and all nodes from right model
    auto right_nodes = right->get_nodes();
    auto right_edges = right->get_edges();
    
    std::vector<NodePtr> new_nodes;
    new_nodes.push_back(left);
    new_nodes.insert(new_nodes.end(), right_nodes.begin(), right_nodes.end());
    
    std::vector<Edge> new_edges = right_edges;
    
    // Connect left node to all input nodes of the right model
    auto input_nodes = right->get_input_nodes();
    for (const auto& input_node : input_nodes) {
        new_edges.emplace_back(left, input_node);
    }
    
    return std::make_shared<Model>(new_nodes, new_edges);
}

std::shared_ptr<Model> operator>>(std::shared_ptr<Model> left, NodePtr right) {
    if (!left || !right) {
        throw std::invalid_argument("Cannot connect null model/node");
    }
    
    // Create new model that includes all nodes from left model and the right node
    auto left_nodes = left->get_nodes();
    auto left_edges = left->get_edges();
    
    std::vector<NodePtr> new_nodes = left_nodes;
    new_nodes.push_back(right);
    
    std::vector<Edge> new_edges = left_edges;
    
    // Connect all output nodes of the left model to the right node
    auto output_nodes = left->get_output_nodes();
    for (const auto& output_node : output_nodes) {
        new_edges.emplace_back(output_node, right);
    }
    
    return std::make_shared<Model>(new_nodes, new_edges);
}

std::shared_ptr<Model> operator&(std::shared_ptr<Model> left, std::shared_ptr<Model> right) {
    if (!left || !right) {
        throw std::invalid_argument("Cannot merge null models");
    }
    
    // Create new model with combined nodes and edges
    auto left_nodes = left->get_nodes();
    auto right_nodes = right->get_nodes();
    auto left_edges = left->get_edges();
    auto right_edges = right->get_edges();
    
    std::vector<NodePtr> combined_nodes;
    combined_nodes.insert(combined_nodes.end(), left_nodes.begin(), left_nodes.end());
    combined_nodes.insert(combined_nodes.end(), right_nodes.begin(), right_nodes.end());
    
    std::vector<Edge> combined_edges;
    combined_edges.insert(combined_edges.end(), left_edges.begin(), left_edges.end());
    combined_edges.insert(combined_edges.end(), right_edges.begin(), right_edges.end());
    
    return std::make_shared<Model>(combined_nodes, combined_edges);
}

} // namespace reservoircpp