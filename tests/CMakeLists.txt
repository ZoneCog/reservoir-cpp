# Find or download Catch2 for testing
find_package(Catch2 3 QUIET)

if(NOT Catch2_FOUND)
    # Download Catch2 using FetchContent
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# Create test executable
add_executable(reservoircpp_tests
    test_main.cpp
    test_version.cpp
    test_types.cpp
    test_utils.cpp
    test_activations.cpp
    test_node.cpp
    test_matrix_generators.cpp
    test_reservoir.cpp
    test_readout.cpp
    test_observables.cpp
    test_datasets.cpp
    test_stage5.cpp
)

target_link_libraries(reservoircpp_tests 
    PRIVATE 
    reservoircpp 
    Catch2::Catch2WithMain
)

# Add tests to CTest
include(Catch)
catch_discover_tests(reservoircpp_tests)