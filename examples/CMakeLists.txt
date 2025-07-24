# Examples CMakeLists.txt

# Simple example demonstrating Stage 1 capabilities
add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example reservoircpp)

# Stage 2 example demonstrating reservoir computing capabilities
add_executable(stage2_example stage2_example.cpp)
target_link_libraries(stage2_example reservoircpp)

# Stage 3 & 4 example demonstrating observables and datasets
add_executable(stage3_stage4_example stage3_stage4_example.cpp)
target_link_libraries(stage3_stage4_example reservoircpp)

# Stage 5 example demonstrating advanced features
add_executable(stage5_example stage5_example.cpp)
target_link_libraries(stage5_example reservoircpp)

# Complete tutorial demonstrating full workflow (Stage 6)
add_executable(complete_tutorial complete_tutorial.cpp)
target_link_libraries(complete_tutorial reservoircpp)

# Practical tutorial with safer examples (Stage 6)
add_executable(practical_tutorial practical_tutorial.cpp)
target_link_libraries(practical_tutorial reservoircpp)

# Stage 6 tutorial - Examples and Documentation
add_executable(stage6_tutorial stage6_tutorial.cpp)
target_link_libraries(stage6_tutorial reservoircpp)

# Stage 7 tutorial - Testing and Quality Assurance
add_executable(stage7_tutorial stage7_tutorial.cpp)
target_link_libraries(stage7_tutorial reservoircpp)

# IntrinsicPlasticity example demonstrating IP reservoir functionality
add_executable(intrinsic_plasticity_example intrinsic_plasticity_example.cpp)
target_link_libraries(intrinsic_plasticity_example reservoircpp)

# NVAR example demonstrating Nonlinear Vector Autoregressive functionality
add_executable(nvar_example nvar_example.cpp)
target_link_libraries(nvar_example reservoircpp)

message(STATUS "Building examples")