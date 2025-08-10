#!/bin/bash
# python_cpp_mapping_analysis.sh
# Script to systematically verify Python to C++ functionality mapping

echo "=== Python to C++ Functionality Mapping Analysis ==="
echo "Date: $(date)"
echo ""

# Function to check if functionality is implemented in C++
check_cpp_implementation() {
    local module=$1
    local description=$2
    echo "Checking: $description"
    
    # Search for related implementations in C++ headers and source files
    found_headers=$(find include/ -name "*.hpp" -exec grep -l "$module" {} \; 2>/dev/null | wc -l)
    found_sources=$(find src/ -name "*.cpp" -exec grep -l "$module" {} \; 2>/dev/null | wc -l)
    
    if [ $found_headers -gt 0 ] || [ $found_sources -gt 0 ]; then
        echo "  ✅ IMPLEMENTED ($found_headers headers, $found_sources sources)"
        return 0
    else
        echo "  ❌ NOT FOUND"
        return 1
    fi
}

echo "## Core Modules Analysis"
echo "========================"

# Core Python modules to check
declare -A CORE_MODULES=(
    ["activationsfunc"]="Activation functions (sigmoid, tanh, relu, etc.)"
    ["mat_gen"]="Matrix generation utilities"
    ["node"]="Base Node class implementation"
    ["model"]="Model class for computational graphs" 
    ["ops"]="Node operations (link, merge, etc.)"
    ["observables"]="Performance metrics and observables"
    ["type"]="Type definitions and utilities"
)

implemented_count=0
total_count=${#CORE_MODULES[@]}

for module in "${!CORE_MODULES[@]}"; do
    check_cpp_implementation "$module" "${CORE_MODULES[$module]}"
    if [ $? -eq 0 ]; then
        ((implemented_count++))
    fi
    echo ""
done

echo "Core Modules: $implemented_count/$total_count implemented"
echo ""

echo "## Node Types Analysis"
echo "======================"

# Node types from reservoirpy/nodes/
declare -A NODE_TYPES=(
    ["reservoir"]="Reservoir computing nodes"
    ["esn"]="Echo State Network implementation"  
    ["ridge"]="Ridge regression readout"
    ["force"]="FORCE learning readout"
    ["rls"]="Recursive Least Squares readout"
    ["lms"]="Least Mean Squares readout"
    ["activations"]="Activation function nodes"
    ["concat"]="Concatenation nodes"
    ["delay"]="Delay line nodes"
    ["intrinsic_plasticity"]="Intrinsic plasticity nodes"
    ["nvar"]="Nonlinear Vector AutoRegression"
)

node_implemented_count=0
node_total_count=${#NODE_TYPES[@]}

for node in "${!NODE_TYPES[@]}"; do
    check_cpp_implementation "$node" "${NODE_TYPES[$node]}"
    if [ $? -eq 0 ]; then
        ((node_implemented_count++))
    fi
    echo ""
done

echo "Node Types: $node_implemented_count/$node_total_count implemented"
echo ""

echo "## Dataset Analysis"
echo "==================="

# Check for dataset implementations
declare -A DATASETS=(
    ["mackey_glass"]="Mackey-Glass time series"
    ["lorenz"]="Lorenz attractor dataset"
    ["henon"]="Henon map dataset"
    ["narma"]="NARMA time series"
    ["doublescroll"]="Double scroll attractor"
    ["rossler"]="Rössler attractor"
)

dataset_implemented_count=0
dataset_total_count=${#DATASETS[@]}

for dataset in "${!DATASETS[@]}"; do
    check_cpp_implementation "$dataset" "${DATASETS[$dataset]}"
    if [ $? -eq 0 ]; then
        ((dataset_implemented_count++))
    fi
    echo ""
done

echo "Datasets: $dataset_implemented_count/$dataset_total_count implemented"
echo ""

echo "## Summary"
echo "=========="
echo "Core Modules: $implemented_count/$total_count"
echo "Node Types: $node_implemented_count/$node_total_count" 
echo "Datasets: $dataset_implemented_count/$dataset_total_count"
echo ""

total_features=$((total_count + node_total_count + dataset_total_count))
total_implemented=$((implemented_count + node_implemented_count + dataset_implemented_count))

echo "OVERALL: $total_implemented/$total_features features implemented"

if [ $total_implemented -eq $total_features ]; then
    echo "🎉 All functionality appears to be implemented in C++!"
    echo "Safe to proceed with Python file migration to TO_REMOVE/"
else
    echo "⚠️  Some functionality may be missing. Manual verification recommended."
fi
