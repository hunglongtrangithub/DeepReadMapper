cmake_minimum_required(VERSION 3.13)
project(DeepAlignerCPU VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
add_compile_options(-Wall)

# Find OpenVINO
if(NOT DEFINED OpenVINO_DIR)
    set(OpenVINO_DIR "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/openvino/cmake")
endif()

find_package(OpenVINO REQUIRED)

# Include directories
include_directories(
    includes
)

# Define source files
set(SOURCES
    src/main.cpp
    src/utils.cpp
    src/vectorize.cpp
    src/fast_model.cpp
    src/preprocess.cpp
    src/tok2index.cpp
)

# Create executable
add_executable(pipeline ${SOURCES})

# Link OpenVINO libraries
target_link_libraries(pipeline openvino::runtime)
