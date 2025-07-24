# CMakeLists.txt for src directory
# Build the core library with Stage 2 components

# Create the core library
add_library(reservoircpp_core STATIC
    matrix_generators.cpp
    reservoir.cpp
    readout.cpp
    observables.cpp
    datasets.cpp
    experimental.cpp
    compat.cpp
    hyper.cpp
    plotting.cpp
    # gpu.cpp  # Temporarily disabled due to compilation issues
)

# Set include directories
target_include_directories(reservoircpp_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

# Link with Eigen
target_link_libraries(reservoircpp_core PUBLIC Eigen3::Eigen)

# Set C++ standard
target_compile_features(reservoircpp_core PUBLIC cxx_std_17)