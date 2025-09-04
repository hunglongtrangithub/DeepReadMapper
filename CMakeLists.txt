cmake_minimum_required(VERSION 3.14)
project(DeepAlignerCPU VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compiler flags - march=native already includes AVX2
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
add_subdirectory(external/cnpy)

# Locate FAISS for HNSW-PQ
# Find FAISS in conda
find_path(FAISS_INCLUDE_DIR
    NAMES faiss/IndexHNSW.h faiss/Index.h
    PATHS $ENV{CONDA_PREFIX}/include
    NO_DEFAULT_PATH
)

find_library(FAISS_LIBRARY
    NAMES faiss
    PATHS $ENV{CONDA_PREFIX}/lib
    NO_DEFAULT_PATH
)

if(NOT FAISS_INCLUDE_DIR OR NOT FAISS_LIBRARY)
    message(FATAL_ERROR "FAISS not found in conda! Please run: conda install faiss-cpu")
endif()

# Print FAISS info
message(STATUS "FAISS include: ${FAISS_INCLUDE_DIR}")
message(STATUS "FAISS library: ${FAISS_LIBRARY}")


# Include directories
include_directories(
    /usr/local/include
    includes
    includes/inference
    includes/utils
    includes/hnswlib_dir # Original HNSW
    # includes/hnswm # Minh's HNSW
    # includes/gann_hnsw # GANN's HNSW
    includes/hnswpq # Faiss HNSWPQ
    external/cereal/include
    external/cnpy
)


# Define source files for pipeline executable
set(PIPELINE_SOURCES
    src/main.cpp

    # For Inference
    src/utils/parse_inputs.cpp
    src/utils/utils.cpp
    src/utils/tok2index.cpp
    src/utils/post_processor.cpp

    src/inference/vectorize.cpp
    src/inference/fast_model.cpp
    src/inference/preprocess.cpp

    # For HNSW search
    # src/hnswlib_dir/search.cpp
    # src/hnswm/hnsw.cpp
    # src/hnswm/bruteforce.cpp
    # src/gann_hnsw/gann_hnsw.cpp
    src/hnswpq/search.cpp
)


# Create pipeline executable (for inference + HNSW search)
add_executable(pipeline ${PIPELINE_SOURCES})

target_include_directories(pipeline PRIVATE ${FAISS_INCLUDE_DIR})

# Link libraries for pipeline
target_link_libraries(pipeline
    openvino::runtime
    OpenMP::OpenMP_CXX
    hnswlib
    cnpy
    ${FAISS_LIBRARY}
)

# Define source files for index executable
set(INDEX_SOURCES
    # For Inference
    src/utils/parse_inputs.cpp
    src/utils/utils.cpp
    src/utils/tok2index.cpp
    src/inference/vectorize.cpp
    src/inference/fast_model.cpp
    src/inference/preprocess.cpp
    # For HNSW indexing
    src/hnswlib_dir/index.cpp
)

# Create index executable
add_executable(index ${INDEX_SOURCES})

# Link libraries for index
target_link_libraries(index
    openvino::runtime
    OpenMP::OpenMP_CXX
    hnswlib
    cnpy
)

# DEFINE executable for testing inference only
set(INFERENCE_SRCS
    src/utils/parse_inputs.cpp
    src/utils/utils.cpp
    src/utils/tok2index.cpp
    src/inference/test_inference.cpp
    src/inference/vectorize.cpp
    src/inference/fast_model.cpp
    src/inference/preprocess.cpp
)

add_executable(inference ${INFERENCE_SRCS})
target_link_libraries(inference
    openvino::runtime
)

# DEFINE executable for testing HNSW from Minh's code
# set(HNSWM_BUILD_SRCS
#     # For Inference
#     src/utils/parse_inputs.cpp
#     src/utils/utils.cpp
#     src/utils/tok2index.cpp
#     src/inference/vectorize.cpp
#     src/inference/fast_model.cpp
#     src/inference/preprocess.cpp

#     # For HNSW indexing
#     src/hnswm/index.cpp
#     src/hnswm/hnsw.cpp
#     src/hnswm/bruteforce.cpp
# )

# add_executable(hnswm_build ${HNSWM_BUILD_SRCS})
# target_link_libraries(hnswm_build
#     openvino::runtime
#     OpenMP::OpenMP_CXX
#     cnpy
# )

# set(HNSWM_SEARCH_SRCS
#     # For Inference
#     src/utils/parse_inputs.cpp
#     src/utils/utils.cpp
#     src/utils/tok2index.cpp
#     src/inference/vectorize.cpp
#     src/inference/fast_model.cpp
#     src/inference/preprocess.cpp

#     # For HNSW search
#     src/hnswm/search.cpp
#     src/hnswm/hnsw.cpp
#     src/hnswm/bruteforce.cpp
# )

# add_executable(hnswm_search ${HNSWM_SEARCH_SRCS})
# target_link_libraries(hnswm_search
#     openvino::runtime
#     OpenMP::OpenMP_CXX
#     cnpy
# )

# DEFINE executable for testing HNSWLib search
# set(HNSWLIB_SEARCH_SRCS
#     src/hnswlib_dir/test_search.cpp
#     # For Inference
#     src/utils/parse_inputs.cpp
#     src/utils/utils.cpp
#     src/utils/tok2index.cpp
#     src/inference/vectorize.cpp
#     src/inference/fast_model.cpp
#     src/inference/preprocess.cpp

#     # For HNSW search
#     src/hnswlib_dir/search.cpp
# )

# add_executable(hnswlib_search ${HNSWLIB_SEARCH_SRCS})
# target_link_libraries(hnswlib_search
#     openvino::runtime
#     OpenMP::OpenMP_CXX
#     hnswlib
#     cnpy
# )

# DEFINE executable for testing GANN's HNSW indexing
# set(GANN_INDEX_SRCS
#     src/gann_hnsw/test_gann_build.cpp
#     # For Inference
#     src/utils/parse_inputs.cpp
#     src/utils/utils.cpp
#     src/utils/tok2index.cpp
#     src/inference/vectorize.cpp
#     src/inference/fast_model.cpp
#     src/inference/preprocess.cpp

#     # For GANN's HNSW indexing
#     src/gann_hnsw/gann_hnsw.cpp
# )

# add_executable(gann_index ${GANN_INDEX_SRCS})

# target_link_libraries(gann_index
#     OpenMP::OpenMP_CXX
#     openvino::runtime
#     cnpy
# )

# DEFINE executable for testing GANN's HNSW search
# set(GANN_SEARCH_SRCS
#     src/gann_hnsw/test_gann_search.cpp
#     # For Inference
#     src/utils/parse_inputs.cpp
#     src/utils/utils.cpp
#     src/utils/tok2index.cpp
#     src/inference/vectorize.cpp
#     src/inference/fast_model.cpp
#     src/inference/preprocess.cpp

#     # For GANN's HNSW search
#     src/gann_hnsw/gann_hnsw.cpp
# )

# add_executable(gann_search ${GANN_SEARCH_SRCS})
# target_link_libraries(gann_search
#     OpenMP::OpenMP_CXX
#     openvino::runtime
#     cnpy
# )

# DEFINE executable for testing HNSW with FAISS's IndexHNSWPQ
set(HNSWPQ_INDEX_SRCS
    src/utils/parse_inputs.cpp
    src/utils/utils.cpp
    src/utils/tok2index.cpp

    src/hnswpq/index.cpp

    src/inference/vectorize.cpp
    src/inference/fast_model.cpp
    src/inference/preprocess.cpp
)

add_executable(hnswpq_index ${HNSWPQ_INDEX_SRCS})
target_include_directories(hnswpq_index PRIVATE ${FAISS_INCLUDE_DIR})
target_link_libraries(hnswpq_index
    openvino::runtime
    OpenMP::OpenMP_CXX
    cnpy
    ${FAISS_LIBRARY}
)

# DEFINE executable for testing HNSW with FAISS's IndexHNSWPQ
set(HNSWPQ_SEARCH_SRCS
    src/utils/parse_inputs.cpp
    src/utils/utils.cpp
    src/utils/tok2index.cpp
    src/hnswpq/test_search.cpp

    src/inference/vectorize.cpp
    src/inference/fast_model.cpp
    src/inference/preprocess.cpp

    src/hnswpq/search.cpp
)

add_executable(hnswpq_search ${HNSWPQ_SEARCH_SRCS})
target_include_directories(hnswpq_search PRIVATE ${FAISS_INCLUDE_DIR})
target_link_libraries(hnswpq_search
    openvino::runtime
    OpenMP::OpenMP_CXX
    cnpy
    ${FAISS_LIBRARY}
)

# DEFINE executable for generating ref.txt
# set(GENERATE_REF_SRCS
#     src/utils/parse_inputs.cpp
#     src/utils/utils.cpp
#     src/gen_ref.cpp
# )

# add_executable(gen_ref ${GENERATE_REF_SRCS})

# target_link_libraries(gen_ref
#     cnpy
# )

# DEFINE executable for SW calc
set(SW_SRCS
    src/test_sw_score.cpp
    src/utils/post_processor.cpp
    src/utils/utils.cpp
)
add_executable(calc_sw ${SW_SRCS})

target_link_libraries(calc_sw
    cnpy
)