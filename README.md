# Optimized DeepAligner CPU

This repository is an opitmized version of DeepAligner CPU Pipeline to locate similar genes of a given input.

## Optimization

- Migrate from Python Binding to native C++.
- Implement several methods to improve multi-threading performance, including privatization, shared memory, better locks/semaphores, ...

## Installation

1. Create conda environment

    ```bash
    conda create -f environment.yml
    conda activate DeepAligner
    ```

2. Install external libraries

    ```bash
    bash setup_submodule.sh
    ```

3. Build the project

    ```bash
    mkdir build && cd build
    cmake ..
    make -j32
    ```
