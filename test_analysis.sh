#!/bin/bash
# Test summary script

cd /workspaces/reservoir-cpp/build

echo "=== ReservoirCpp Test Analysis ==="
echo "Date: $(date)"
echo

# Run tests and capture results
echo "Running tests..."
ctest --output-on-failure > test_results.log 2>&1

# Extract summary
echo "=== TEST SUMMARY ==="
if grep -q "100% tests passed" test_results.log; then
    echo "✅ ALL TESTS PASSED!"
    total_tests=$(grep -o "[0-9]*/" test_results.log | tail -1 | sed 's|/||')
    echo "Total tests: $total_tests"
else
    echo "❌ Some tests failed"
    passed=$(grep -o "[0-9]* Passed" test_results.log | tail -1 | grep -o "[0-9]*")
    failed=$(grep -o "[0-9]* Failed" test_results.log | tail -1 | grep -o "[0-9]*")
    total=$(grep -o "[0-9]*/" test_results.log | tail -1 | sed 's|/||')
    echo "Passed: $passed/$total"
    echo "Failed: $failed/$total"
    
    echo
    echo "Failed tests:"
    grep -A2 -B2 "Failed" test_results.log
fi

echo
echo "=== BUILD STATUS ==="
if [ -f reservoircpp_tests ] && [ -f tests/reservoircpp_tests ]; then
    echo "✅ Test executable built successfully"
else
    echo "❌ Test executable missing"
fi

# Check examples
echo
echo "=== EXAMPLES STATUS ==="
example_count=$(find examples/ -name "*_example" -o -name "*tutorial*" 2>/dev/null | wc -l)
echo "Built examples: $example_count"

echo
echo "=== RECENT TEST LOG (last 50 lines) ==="
tail -50 test_results.log
