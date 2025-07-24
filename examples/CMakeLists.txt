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

message(STATUS "Building examples")