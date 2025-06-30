cmake_minimum_required(VERSION 3.13)
project(DeepAlignerCPU VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compiler flags (similar to your existing projects)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
add_compile_options(-Wall)

# Include directories
include_directories(
    includes
)


# Define source files
set(SOURCES
    src/main.cpp
    src/vectorize.cpp
)

# Create executable
add_executable(pipeline ${SOURCES})
