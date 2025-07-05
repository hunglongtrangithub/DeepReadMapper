cmake_minimum_required(VERSION 3.14)
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

# Add hnswlib
add_subdirectory(external/hnswlib)

# Include directories
include_directories(
    includes
    external/hnswlib/hnswlib
)

# Define source files for pipeline executable
set(PIPELINE_SOURCES
    src/main.cpp
    # For Inference
    src/utils/utils.cpp
    src/utils/tok2index.cpp
    src/inference/vectorize.cpp
    src/inference/fast_model.cpp
    src/inference/preprocess.cpp
    # For HNSW search
    src/hnsw_dir/search.cpp
)

# Create pipeline executable
add_executable(pipeline ${PIPELINE_SOURCES})

# Link libraries for pipeline
target_link_libraries(pipeline
    openvino::runtime
    hnswlib
)

# Define source files for index executable
set(INDEX_SOURCES
    # For Inference
    src/utils/utils.cpp
    src/utils/tok2index.cpp
    src/inference/vectorize.cpp
    src/inference/fast_model.cpp
    src/inference/preprocess.cpp
    # For HNSW indexing
    src/hnsw_dir/index.cpp
)

# Create index executable
add_executable(index ${INDEX_SOURCES})

# Link libraries for index
target_link_libraries(index
    openvino::runtime
    hnswlib
)
