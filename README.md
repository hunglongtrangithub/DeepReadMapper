# Optimized DeepAligner CPU

This repository is an opitmized version of DeepAligner CPU Pipeline to locate similar genes of a given input.

## Optimization

- Migrate from Python Binding to native C++.
- Implement several methods to improve multi-threading performance, including privatization, shared memory, better locks/semaphores, ...
