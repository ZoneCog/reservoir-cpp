/**
 * @file ops.hpp
 * @brief Node operations - link, merge, feedback for ReservoirCpp
 * 
 * This file contains the C++ port of the Python ops.py,
 * implementing high-level operations for connecting nodes and models.
 * 
 * Operations on Node and Model:
 * - link: Link Nodes into a Model
 * - link_feedback: Link Nodes through feedback connections  
 * - merge: Merge Models
 */

#ifndef RESERVOIRCPP_OPS_HPP
#define RESERVOIRCPP_OPS_HPP

#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

#include "reservoircpp/types.hpp"
#include "reservoircpp/node.hpp"
#include "reservoircpp/model.hpp"

namespace reservoircpp {

// Forward declarations
class Concat;

namespace ops {

/**
 * @brief Link two Node instances to form a Model instance
 * 
 * node1 output will be used as input for node2 in the created model.
 * This is similar to a function composition operation:
 * 
 * model(x) = (node1 ∘ node2)(x) = node2(node1(x))
 * 
 * You can also perform this operation using the >> operator:
 * 
 *     auto model = node1 >> node2;
 * 
 * Or using this function:
 * 
 *     auto model = link(node1, node2);
 * 
 * @param node1 First node (output feeds into node2)
 * @param node2 Second node (receives input from node1)  
 * @param name Name for the chaining Model (optional)
 * @return Shared pointer to Model instance chaining the nodes
 * @throws std::invalid_argument if nodes are null
 * @throws std::runtime_error for dimension mismatches between connected nodes
 */
std::shared_ptr<Model> link(NodePtr node1, NodePtr node2, const std::string& name = "");

/**
 * @brief Link a node to a model
 * 
 * @param node Node to link (output feeds into model inputs)
 * @param model Model to receive input from node
 * @param name Name for the resulting Model (optional)
 * @return Shared pointer to Model instance
 */
std::shared_ptr<Model> link(NodePtr node, std::shared_ptr<Model> model, const std::string& name = "");

/**
 * @brief Link a model to a node
 * 
 * @param model Model whose outputs feed into node
 * @param node Node to receive input from model outputs
 * @param name Name for the resulting Model (optional)
 * @return Shared pointer to Model instance
 */
std::shared_ptr<Model> link(std::shared_ptr<Model> model, NodePtr node, const std::string& name = "");

/**
 * @brief Link multiple nodes to a single node
 * 
 * All nodes in the vector will be connected to a Concat node,
 * and the Concat node will be linked to the output node.
 * 
 * @param input_nodes Vector of input nodes to concatenate
 * @param output_node Output node to receive concatenated input
 * @param name Name for the resulting Model (optional)
 * @return Shared pointer to Model instance
 */
std::shared_ptr<Model> link(const std::vector<NodePtr>& input_nodes, NodePtr output_node, const std::string& name = "");

/**
 * @brief Link a node to multiple nodes
 * 
 * The input node will be connected to all nodes in the vector.
 * Each output node receives the same input.
 * 
 * @param input_node Input node to broadcast to multiple outputs
 * @param output_nodes Vector of output nodes to receive input
 * @param name Name for the resulting Model (optional)
 * @return Shared pointer to Model instance
 */
std::shared_ptr<Model> link(NodePtr input_node, const std::vector<NodePtr>& output_nodes, const std::string& name = "");

/**
 * @brief Link multiple nodes to multiple nodes
 * 
 * All input nodes will be connected to a Concat node,
 * and the Concat node will be linked to all output nodes.
 * 
 * @param input_nodes Vector of input nodes to concatenate
 * @param output_nodes Vector of output nodes to receive concatenated input
 * @param name Name for the resulting Model (optional)
 * @return Shared pointer to Model instance
 */
std::shared_ptr<Model> link(const std::vector<NodePtr>& input_nodes, const std::vector<NodePtr>& output_nodes, const std::string& name = "");

/**
 * @brief Create a feedback connection between nodes
 * 
 * Feedback nodes will be called at runtime using data from the previous call.
 * This is not an in-place operation by default. This function will copy the node
 * and then sets the copy's feedback attribute.
 * 
 * You can also perform this operation using the << operator:
 * 
 *     auto node_with_feedback = node << feedback_node;
 * 
 * Which means that a feedback connection is now created. The forward function 
 * of node depends on the previous output of feedback_node:
 * 
 * node(x_t) = node(x_t, feedback_node(x_{t-1}))
 * 
 * @param node Node receiving feedback
 * @param feedback Node or Model sending feedback
 * @param inplace If true, modifies node in-place instead of creating copy
 * @param name Name of the copy if inplace is false (optional)
 * @return Node instance taking feedback from feedback node
 * @throws std::invalid_argument if nodes are null or invalid
 */
NodePtr link_feedback(NodePtr node, NodePtr feedback, bool inplace = false, const std::string& name = "");

/**
 * @brief Create feedback connection with multiple feedback nodes
 * 
 * All feedback nodes will be connected to a Concat node,
 * and the Concat node will provide feedback to the main node.
 * 
 * @param node Node receiving feedback
 * @param feedback_nodes Vector of nodes providing feedback
 * @param inplace If true, modifies node in-place instead of creating copy
 * @param name Name of the copy if inplace is false (optional)
 * @return Node instance taking feedback
 */
NodePtr link_feedback(NodePtr node, const std::vector<NodePtr>& feedback_nodes, bool inplace = false, const std::string& name = "");

/**
 * @brief Create feedback connection with a model
 * 
 * @param node Node receiving feedback
 * @param feedback_model Model providing feedback
 * @param inplace If true, modifies node in-place instead of creating copy
 * @param name Name of the copy if inplace is false (optional)
 * @return Node instance taking feedback
 */
NodePtr link_feedback(NodePtr node, std::shared_ptr<Model> feedback_model, bool inplace = false, const std::string& name = "");

/**
 * @brief Merge different Model or Node instances into a single Model
 * 
 * Node instances contained in the models to merge will be gathered in a single model,
 * along with all previously defined connections between them.
 * 
 * You can also perform this operation using the & operator:
 * 
 *     auto model = (node1 >> node2) & (node1 >> node3);
 * 
 * This is equivalent to:
 * 
 *     auto model = merge((node1 >> node2), (node1 >> node3));
 * 
 * @param model First node or model to merge
 * @param other_models Additional models to merge
 * @param name Name of the resulting Model (optional)
 * @return New Model instance containing all nodes and edges
 * @throws std::invalid_argument if any model is null
 */
std::shared_ptr<Model> merge(std::shared_ptr<Model> model, const std::vector<std::shared_ptr<Model>>& other_models, const std::string& name = "");

/**
 * @brief Merge two models
 * 
 * @param model1 First model to merge
 * @param model2 Second model to merge
 * @param name Name of the resulting Model (optional)
 * @return New Model instance
 */
std::shared_ptr<Model> merge(std::shared_ptr<Model> model1, std::shared_ptr<Model> model2, const std::string& name = "");

/**
 * @brief Merge a model with a node
 * 
 * @param model Model to merge
 * @param node Node to merge
 * @param name Name of the resulting Model (optional)
 * @return New Model instance
 */
std::shared_ptr<Model> merge(std::shared_ptr<Model> model, NodePtr node, const std::string& name = "");

/**
 * @brief Merge two nodes into a model
 * 
 * @param node1 First node to merge
 * @param node2 Second node to merge
 * @param name Name of the resulting Model (optional)
 * @return New Model instance
 */
std::shared_ptr<Model> merge(NodePtr node1, NodePtr node2, const std::string& name = "");

} // namespace ops

} // namespace reservoircpp

#endif // RESERVOIRCPP_OPS_HPP