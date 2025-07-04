# Examples CMakeLists.txt

# Simple example demonstrating Stage 1 capabilities
add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example reservoircpp)

# Stage 2 example demonstrating reservoir computing capabilities
add_executable(stage2_example stage2_example.cpp)
target_link_libraries(stage2_example reservoircpp)

message(STATUS "Building examples")