#!/bin/bash
# Component analysis script

echo "=== ReservoirCpp Component Analysis ==="
echo

echo "=== HEADER FILES ==="
cd /workspaces/reservoir-cpp
find include/reservoircpp -name "*.hpp" | sort

echo
echo "=== SOURCE FILES ==="
find src -name "*.cpp" | sort

echo
echo "=== TEST FILES ==="
find tests -name "*.cpp" | sort

echo
echo "=== EXAMPLE FILES ==="
find examples -name "*.cpp" | sort

echo
echo "=== CORE CLASSES IMPLEMENTED ==="
echo "Looking for main classes:"
grep -r "^class.*:" include/reservoircpp/ | grep -v "//" | sed 's|.*:class ||' | sed 's| :.*||' | sort | uniq

echo
echo "=== FUNCTIONS/METHODS COUNT ==="
echo "Header files line count:"
find include/reservoircpp -name "*.hpp" -exec wc -l {} + | tail -1

echo "Source files line count:"
find src -name "*.cpp" -exec wc -l {} + | tail -1

echo "Test files line count:"
find tests -name "*.cpp" -exec wc -l {} + | tail -1

echo
echo "=== KEY FUNCTIONALITY CHECK ==="
echo "Checking for key implementations:"

echo "- Reservoir classes:"
grep -r "class.*Reservoir" include/ | wc -l

echo "- Node implementations:"
grep -r "class.*Node" include/ | wc -l

echo "- Activation functions:"
grep -r "inline Matrix.*(" include/reservoircpp/activations.hpp | wc -l

echo "- Dataset functions:"
grep -r "inline.*(" include/reservoircpp/datasets.hpp | wc -l

echo
echo "=== POTENTIAL ISSUES ==="
echo "Checking for potential issues:"

echo "- TODO/FIXME comments:"
grep -r "TODO\|FIXME" include/ src/ | wc -l

echo "- Exception handling:"
grep -r "throw\|except" src/ | wc -l

echo "- Memory management (smart pointers):"
grep -r "shared_ptr\|unique_ptr" include/ src/ | wc -l
