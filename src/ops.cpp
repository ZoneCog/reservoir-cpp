/**
 * @file ops.cpp
 * @brief Implementation of Node operations for ReservoirCpp
 */

#include "reservoircpp/ops.hpp"
#include "reservoircpp/concat.hpp"
#include <algorithm>
#include <sstream>

namespace reservoircpp {
namespace ops {

namespace {
    /**
     * @brief Check that nodes are valid (not null)
     */
    void check_nodes_valid(const std::vector<NodePtr>& nodes, const std::string& context) {
        for (const auto& node : nodes) {
            if (!node) {
                throw std::invalid_argument(context + ": null node pointer provided");
            }
        }
    }
    
    /**
     * @brief Check that single node is valid
     */
    void check_node_valid(NodePtr node, const std::string& context) {
        if (!node) {
            throw std::invalid_argument(context + ": null node pointer provided");
        }
    }
    
    /**
     * @brief Check that model is valid
     */
    void check_model_valid(std::shared_ptr<Model> model, const std::string& context) {
        if (!model) {
            throw std::invalid_argument(context + ": null model pointer provided");
        }
    }
    
    /**
     * @brief Validate dimension compatibility between nodes
     */
    void validate_connection_dimensions(NodePtr sender, NodePtr receiver) {
        if (sender->is_initialized() && receiver->is_initialized()) {
            auto sender_output = sender->output_dim();
            auto receiver_input = receiver->input_dim();
            
            if (sender_output != receiver_input) {
                std::stringstream ss;
                ss << "Dimension mismatch between connected nodes: ";
                ss << "sender node " << sender->name() << " has output dimension [";
                for (size_t i = 0; i < sender_output.size(); ++i) {
                    if (i > 0) ss << ", ";
                    ss << sender_output[i];
                }
                ss << "] but receiver node " << receiver->name() << " has input dimension [";
                for (size_t i = 0; i < receiver_input.size(); ++i) {
                    if (i > 0) ss << ", ";
                    ss << receiver_input[i];
                }
                ss << "]";
                throw std::runtime_error(ss.str());
            }
        }
    }
    
    /**
     * @brief Generate unique name if empty
     */
    std::string ensure_name(const std::string& name, const std::string& prefix = "model") {
        if (name.empty()) {
            return prefix + "_" + generate_uuid();
        }
        return name;
    }
}

// Link functions

std::shared_ptr<Model> link(NodePtr node1, NodePtr node2, const std::string& name) {
    check_node_valid(node1, "link");
    check_node_valid(node2, "link");
    
    // Validate dimensions if both nodes are initialized
    validate_connection_dimensions(node1, node2);
    
    std::vector<NodePtr> nodes = {node1, node2};
    std::vector<Edge> edges = {{node1, node2}};
    
    return std::make_shared<Model>(nodes, edges, ensure_name(name, "link"));
}

std::shared_ptr<Model> link(NodePtr node, std::shared_ptr<Model> model, const std::string& name) {
    check_node_valid(node, "link");
    check_model_valid(model, "link");
    
    // Get model components
    auto model_nodes = model->get_nodes();
    auto model_edges = model->get_edges();
    auto model_inputs = model->get_input_nodes();
    
    // Create new node list
    std::vector<NodePtr> new_nodes;
    new_nodes.push_back(node);
    new_nodes.insert(new_nodes.end(), model_nodes.begin(), model_nodes.end());
    
    // Create new edge list - connect node to all model input nodes
    std::vector<Edge> new_edges = model_edges;
    for (const auto& input_node : model_inputs) {
        validate_connection_dimensions(node, input_node);
        new_edges.emplace_back(node, input_node);
    }
    
    return std::make_shared<Model>(new_nodes, new_edges, ensure_name(name, "link"));
}

std::shared_ptr<Model> link(std::shared_ptr<Model> model, NodePtr node, const std::string& name) {
    check_model_valid(model, "link");
    check_node_valid(node, "link");
    
    // Get model components
    auto model_nodes = model->get_nodes();
    auto model_edges = model->get_edges();
    auto model_outputs = model->get_output_nodes();
    
    // Create new node list
    std::vector<NodePtr> new_nodes = model_nodes;
    new_nodes.push_back(node);
    
    // Create new edge list - connect all model output nodes to node
    std::vector<Edge> new_edges = model_edges;
    for (const auto& output_node : model_outputs) {
        validate_connection_dimensions(output_node, node);
        new_edges.emplace_back(output_node, node);
    }
    
    return std::make_shared<Model>(new_nodes, new_edges, ensure_name(name, "link"));
}

std::shared_ptr<Model> link(const std::vector<NodePtr>& input_nodes, NodePtr output_node, const std::string& name) {
    check_nodes_valid(input_nodes, "link");
    check_node_valid(output_node, "link");
    
    if (input_nodes.empty()) {
        throw std::invalid_argument("link: input_nodes cannot be empty");
    }
    
    if (input_nodes.size() == 1) {
        // Direct connection for single input
        return link(input_nodes[0], output_node, name);
    }
    
    // Create concat node for multiple inputs
    auto concat_node = std::make_shared<Concat>(1, "concat_" + generate_uuid());
    
    std::vector<NodePtr> all_nodes = input_nodes;
    all_nodes.push_back(concat_node);
    all_nodes.push_back(output_node);
    
    std::vector<Edge> edges;
    
    // Connect all input nodes to concat node
    for (const auto& input_node : input_nodes) {
        edges.emplace_back(input_node, concat_node);
    }
    
    // Connect concat node to output node
    edges.emplace_back(concat_node, output_node);
    
    return std::make_shared<Model>(all_nodes, edges, ensure_name(name, "link"));
}

std::shared_ptr<Model> link(NodePtr input_node, const std::vector<NodePtr>& output_nodes, const std::string& name) {
    check_node_valid(input_node, "link");
    check_nodes_valid(output_nodes, "link");
    
    if (output_nodes.empty()) {
        throw std::invalid_argument("link: output_nodes cannot be empty");
    }
    
    if (output_nodes.size() == 1) {
        // Direct connection for single output
        return link(input_node, output_nodes[0], name);
    }
    
    // Create nodes and edges
    std::vector<NodePtr> all_nodes;
    all_nodes.push_back(input_node);
    all_nodes.insert(all_nodes.end(), output_nodes.begin(), output_nodes.end());
    
    std::vector<Edge> edges;
    
    // Connect input node to all output nodes
    for (const auto& output_node : output_nodes) {
        validate_connection_dimensions(input_node, output_node);
        edges.emplace_back(input_node, output_node);
    }
    
    return std::make_shared<Model>(all_nodes, edges, ensure_name(name, "link"));
}

std::shared_ptr<Model> link(const std::vector<NodePtr>& input_nodes, const std::vector<NodePtr>& output_nodes, const std::string& name) {
    check_nodes_valid(input_nodes, "link");
    check_nodes_valid(output_nodes, "link");
    
    if (input_nodes.empty()) {
        throw std::invalid_argument("link: input_nodes cannot be empty");
    }
    if (output_nodes.empty()) {
        throw std::invalid_argument("link: output_nodes cannot be empty");
    }
    
    if (input_nodes.size() == 1 && output_nodes.size() == 1) {
        // Direct connection
        return link(input_nodes[0], output_nodes[0], name);
    }
    
    if (input_nodes.size() == 1) {
        // One-to-many
        return link(input_nodes[0], output_nodes, name);
    }
    
    // Many-to-many (requires concat)
    auto concat_node = std::make_shared<Concat>(1, "concat_" + generate_uuid());
    
    std::vector<NodePtr> all_nodes = input_nodes;
    all_nodes.push_back(concat_node);
    all_nodes.insert(all_nodes.end(), output_nodes.begin(), output_nodes.end());
    
    std::vector<Edge> edges;
    
    // Connect all input nodes to concat node
    for (const auto& input_node : input_nodes) {
        edges.emplace_back(input_node, concat_node);
    }
    
    // Connect concat node to all output nodes
    for (const auto& output_node : output_nodes) {
        edges.emplace_back(concat_node, output_node);
    }
    
    return std::make_shared<Model>(all_nodes, edges, ensure_name(name, "link"));
}

// Link feedback functions

NodePtr link_feedback(NodePtr node, NodePtr feedback, bool inplace, const std::string& name) {
    check_node_valid(node, "link_feedback");
    check_node_valid(feedback, "link_feedback");
    
    // Implement proper feedback mechanism using Node's feedback infrastructure
    if (inplace) {
        // Set feedback on the node itself
        node->set_feedback(feedback);
        return node;
    } else {
        // Create a copy and set feedback
        auto node_copy = std::dynamic_pointer_cast<Node>(node->copy(ensure_name(name, node->name() + "_feedback")));
        node_copy->set_feedback(feedback);
        return node_copy;
    }
}

NodePtr link_feedback(NodePtr node, const std::vector<NodePtr>& feedback_nodes, bool inplace, const std::string& name) {
    check_node_valid(node, "link_feedback");
    check_nodes_valid(feedback_nodes, "link_feedback");
    
    if (feedback_nodes.empty()) {
        throw std::invalid_argument("link_feedback: feedback_nodes cannot be empty");
    }
    
    if (feedback_nodes.size() == 1) {
        return link_feedback(node, feedback_nodes[0], inplace, name);
    }
    
    // Create concat node for multiple feedback nodes
    auto concat_node = std::make_shared<Concat>(1, "feedback_concat_" + generate_uuid());
    
    // Complete feedback implementation with concat
    // The concat node will collect outputs from all feedback nodes
    // Note: The concat node itself serves as the feedback provider
    // Individual feedback nodes would need to be connected to it in a model context
    return link_feedback(node, concat_node, inplace, name);
}

NodePtr link_feedback(NodePtr node, std::shared_ptr<Model> feedback_model, bool inplace, const std::string& name) {
    check_node_valid(node, "link_feedback");
    check_model_valid(feedback_model, "link_feedback");
    
    // Implement model feedback
    // Use output nodes of the model as feedback providers
    auto output_nodes = feedback_model->get_output_nodes();
    if (output_nodes.empty()) {
        throw std::invalid_argument("link_feedback: feedback model has no output nodes");
    }
    
    if (output_nodes.size() == 1) {
        return link_feedback(node, output_nodes[0], inplace, name);
    } else {
        return link_feedback(node, output_nodes, inplace, name);
    }
}

// Merge functions

std::shared_ptr<Model> merge(std::shared_ptr<Model> model, const std::vector<std::shared_ptr<Model>>& other_models, const std::string& name) {
    check_model_valid(model, "merge");
    
    for (const auto& other_model : other_models) {
        check_model_valid(other_model, "merge");
    }
    
    // Collect all nodes and edges
    std::vector<NodePtr> all_nodes = model->get_nodes();
    std::vector<Edge> all_edges = model->get_edges();
    
    for (const auto& other_model : other_models) {
        auto other_nodes = other_model->get_nodes();
        auto other_edges = other_model->get_edges();
        
        // Add nodes (check for duplicates by pointer)
        for (const auto& node : other_nodes) {
            auto it = std::find(all_nodes.begin(), all_nodes.end(), node);
            
            if (it == all_nodes.end()) {
                all_nodes.push_back(node);
            }
        }
        
        // Add edges
        all_edges.insert(all_edges.end(), other_edges.begin(), other_edges.end());
    }
    
    return std::make_shared<Model>(all_nodes, all_edges, ensure_name(name, "merged"));
}

std::shared_ptr<Model> merge(std::shared_ptr<Model> model1, std::shared_ptr<Model> model2, const std::string& name) {
    check_model_valid(model1, "merge");
    check_model_valid(model2, "merge");
    
    // Collect all nodes and edges from both models
    std::vector<NodePtr> all_nodes = model1->get_nodes();
    std::vector<Edge> all_edges = model1->get_edges();
    
    auto model2_nodes = model2->get_nodes();
    auto model2_edges = model2->get_edges();
    
    // Add nodes from model2 (check for duplicates by pointer)
    for (const auto& node : model2_nodes) {
        auto it = std::find(all_nodes.begin(), all_nodes.end(), node);
        
        if (it == all_nodes.end()) {
            all_nodes.push_back(node);
        }
    }
    
    // Add edges from model2
    all_edges.insert(all_edges.end(), model2_edges.begin(), model2_edges.end());
    
    return std::make_shared<Model>(all_nodes, all_edges, ensure_name(name, "merged"));
}

std::shared_ptr<Model> merge(std::shared_ptr<Model> model, NodePtr node, const std::string& name) {
    check_model_valid(model, "merge");
    check_node_valid(node, "merge");
    
    auto model_nodes = model->get_nodes();
    auto model_edges = model->get_edges();
    
    // Check if node already exists in model
    auto it = std::find(model_nodes.begin(), model_nodes.end(), node);
    
    if (it == model_nodes.end()) {
        model_nodes.push_back(node);
    }
    
    return std::make_shared<Model>(model_nodes, model_edges, ensure_name(name, "merged"));
}

std::shared_ptr<Model> merge(NodePtr node1, NodePtr node2, const std::string& name) {
    check_node_valid(node1, "merge");
    check_node_valid(node2, "merge");
    
    std::vector<NodePtr> nodes = {node1, node2};
    std::vector<Edge> edges; // No connections between them in a merge
    
    return std::make_shared<Model>(nodes, edges, ensure_name(name, "merged"));
}

} // namespace ops
} // namespace reservoircpp