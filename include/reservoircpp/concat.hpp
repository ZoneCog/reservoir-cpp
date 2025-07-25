/**
 * @file concat.hpp
 * @brief Concat node for ReservoirCpp - concatenates multiple inputs
 * 
 * This file contains the C++ implementation of the Concat node,
 * which concatenates vector data along the feature axis.
 */

#ifndef RESERVOIRCPP_CONCAT_HPP
#define RESERVOIRCPP_CONCAT_HPP

#include <vector>
#include "reservoircpp/node.hpp"

namespace reservoircpp {

/**
 * @brief Concatenate vector of data along feature axis
 * 
 * This node is automatically created when a node receives input
 * from more than one node in graph operations.
 */
class Concat : public Node {
public:
    /**
     * @brief Constructor for Concat node
     * 
     * @param axis Concatenation axis (1 for features, 0 for samples)
     * @param name Node name (optional)
     */
    explicit Concat(int axis = 1, const std::string& name = "");
    
    /**
     * @brief Virtual destructor
     */
    virtual ~Concat() = default;
    
    /**
     * @brief Process multiple inputs and concatenate them
     * 
     * @param inputs Vector of input matrices to concatenate
     * @return Concatenated matrix
     */
    Matrix forward_multiple(const std::vector<Matrix>& inputs);
    
    /**
     * @brief Get concatenation axis
     * 
     * @return Concatenation axis
     */
    int get_axis() const;
    
    /**
     * @brief Set concatenation axis
     * 
     * @param axis New concatenation axis
     */
    void set_axis(int axis);

protected:
    /**
     * @brief Forward pass implementation (single input - pass through)
     * 
     * @param input Input matrix
     * @return Output matrix (same as input for single input)
     */
    Matrix forward(const Matrix& input) override;
    
    /**
     * @brief Initialize the concat node
     * 
     * @param x Input data
     * @param y Output data (unused)
     */
    void do_initialize(const Matrix* x, const Matrix* y) override;

private:
    int axis_; ///< Concatenation axis (0 = rows/samples, 1 = cols/features)
};

} // namespace reservoircpp

#endif // RESERVOIRCPP_CONCAT_HPP