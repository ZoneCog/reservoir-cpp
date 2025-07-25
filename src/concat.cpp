/**
 * @file concat.cpp
 * @brief Implementation of Concat node for ReservoirCpp
 */

#include "reservoircpp/concat.hpp"
#include <sstream>

namespace reservoircpp {

Concat::Concat(int axis, const std::string& name)
    : Node(name.empty() ? ("concat_" + generate_uuid()) : name)
    , axis_(axis) {
    
    if (axis_ != 0 && axis_ != 1) {
        throw std::invalid_argument("Concat: axis must be 0 (rows) or 1 (columns)");
    }
    
    // Set hyperparameters
    get_hypers()["axis"] = axis_;
}

Matrix Concat::forward(const Matrix& input) {
    // For single input, just pass through
    return input;
}

Matrix Concat::forward_multiple(const std::vector<Matrix>& inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument("Concat: cannot concatenate empty input list");
    }
    
    if (inputs.size() == 1) {
        return inputs[0];
    }
    
    // Validate all inputs have consistent dimensions
    int rows = inputs[0].rows();
    int total_cols = 0;
    
    if (axis_ == 1) {
        // Concatenate along columns (features)
        for (const auto& input : inputs) {
            if (input.rows() != rows) {
                std::stringstream ss;
                ss << "Concat: inconsistent number of rows. Expected " << rows 
                   << " but got " << input.rows();
                throw std::invalid_argument(ss.str());
            }
            total_cols += input.cols();
        }
        
        // Create result matrix
        Matrix result(rows, total_cols);
        
        // Copy data
        int col_offset = 0;
        for (const auto& input : inputs) {
            result.middleCols(col_offset, input.cols()) = input;
            col_offset += input.cols();
        }
        
        return result;
        
    } else if (axis_ == 0) {
        // Concatenate along rows (samples)
        int cols = inputs[0].cols();
        int total_rows = 0;
        
        for (const auto& input : inputs) {
            if (input.cols() != cols) {
                std::stringstream ss;
                ss << "Concat: inconsistent number of columns. Expected " << cols 
                   << " but got " << input.cols();
                throw std::invalid_argument(ss.str());
            }
            total_rows += input.rows();
        }
        
        // Create result matrix
        Matrix result(total_rows, cols);
        
        // Copy data
        int row_offset = 0;
        for (const auto& input : inputs) {
            result.middleRows(row_offset, input.rows()) = input;
            row_offset += input.rows();
        }
        
        return result;
        
    } else {
        throw std::invalid_argument("Concat: axis must be 0 (rows) or 1 (columns)");
    }
}

void Concat::do_initialize(const Matrix* x, const Matrix* y) {
    if (x != nullptr) {
        // Set input and output dimensions based on first input
        // For concat, output dimensions depend on the number of inputs,
        // which we'll determine at runtime
        set_input_dim({static_cast<int>(x->rows()), static_cast<int>(x->cols())});
        
        // For now, set output dim same as input (will be updated when multiple inputs are processed)
        set_output_dim({static_cast<int>(x->rows()), static_cast<int>(x->cols())});
    }
    
    (void)y; // Suppress unused parameter warning
}

int Concat::get_axis() const {
    return axis_;
}

void Concat::set_axis(int axis) {
    if (axis != 0 && axis != 1) {
        throw std::invalid_argument("Concat: axis must be 0 (rows) or 1 (columns)");
    }
    axis_ = axis;
    get_hypers()["axis"] = axis_;
}

} // namespace reservoircpp