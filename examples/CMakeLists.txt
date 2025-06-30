# Examples CMakeLists.txt

# Simple example demonstrating Stage 0 capabilities
add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example reservoircpp)

# Future examples will be added here as the library develops
# add_executable(basic_reservoir_demo basic_reservoir_demo.cpp)
# target_link_libraries(basic_reservoir_demo reservoircpp)

message(STATUS "Building examples")