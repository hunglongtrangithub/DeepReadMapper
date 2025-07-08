cmake_minimum_required(VERSION 3.14)
project(DeepAlignerCPU VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")
add_compile_options(-Wall)

# Find OpenVINO for inference
if(NOT DEFINED OpenVINO_DIR)
    set(OpenVINO_DIR "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/openvino/cmake")
endif()

find_package(OpenVINO REQUIRED)

# Find openmp for HNSW
# Find OpenMP
find_package(OpenMP REQUIRED)

# Add hnswlib
add_subdirectory(external/hnswlib)

# Include directories
include_directories(
    /usr/local/include
    includes
    includes/inference
    includes/utils
    includes/hnswlib_dir
    includes/hnswm # Minh's HNSW
    external/hnswlib/hnswlib
    external/cereal/include
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
    src/hnswlib_dir/search.cpp
    src/hnswm/hnsw.cpp
    src/hnswm/bruteforce.cpp
)

# Create pipeline executable
add_executable(pipeline ${PIPELINE_SOURCES})

# Link libraries for pipeline
target_link_libraries(pipeline
    openvino::runtime
    hnswlib
    OpenMP::OpenMP_CXX
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
    src/hnswlib_dir/index.cpp
    src/hnswm/hnsw.cpp
    src/hnswm/bruteforce.cpp
)

# Create index executable
add_executable(index ${INDEX_SOURCES})

# Link libraries for index
target_link_libraries(index
    openvino::runtime
    hnswlib
    OpenMP::OpenMP_CXX
)

# DEFINE executable for testing HNSW from Minh's code
set(HNSWM_INDEX_SRCS
    src/hnswm/test_build.cpp
    src/hnswm/hnsw.cpp
    src/hnswm/bruteforce.cpp
)

add_executable(test_hnsw ${HNSWM_INDEX_SRCS})
target_link_libraries(test_hnsw
    OpenMP::OpenMP_CXX
)

set(HNSWM_SEARCH_SRCS
    src/hnswm/test_search.cpp
    src/hnswm/hnsw.cpp
    src/hnswm/bruteforce.cpp
)

add_executable(test_hnsw_search ${HNSWM_SEARCH_SRCS})
target_link_libraries(test_hnsw_search
    OpenMP::OpenMP_CXX
)
