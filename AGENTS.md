## Objective

The objective of this project is to write CUDA code that constructs a **Delaunay Triangulation**, given a set of points in 2-D plane.

A sample code written is Thrust is available in `sample.cu`. This is however just a guideline and the final code should be written in CUDA using explicit device and host pointers in the typical CUDA programming workflow:

1. Load the inputs from host RAM to GPU memory (using CudaMemcpy etc)
2. Launch kernels with appropriate values for blocks_per_grid, threads_per_block, amount of shared memory, streams (if any)
3. Copy the output from device memory to host memory.

A plotting script is given in `plot.py`. Do not try to execute this script as numpy and matplotlib is not available.

Write custom __global__ and __device__ kernels and explicit CUDA runtime API calls (cudaMalloc, cudaMemcpy, cudaFree, streams, and events) for memory management, execution configuration, and synchronization instead of relying on Thrust, focusing directly on project files rather than browsing external references.

The code should be readable and organized.

Complete the objective before moving on to optimizations.

The work should be done in meaningful commits.

**NOTE** - this agent will be running on a system which does not support cuda runtime, so do not run installation commands or execute code. However
nvcc 13.5 is available on the system on which the agent can check for compilation errors.

## Optimizations

The following types of optimizations are in scope for this project:

1. Splitting work into streams which are triggered using events.
2. Using shared memory to improve performance. (use this in appropriate kernels)
3. Minimizing bank conflicts in shared memory. (only a stretch goal)


## Target system and IO

Target architecture: NVIDIA A30 (`sm_80`).

Use `nvcc` (13.5) to verify syntax, types, and compilation; never execute binaries, profilers, or GPU runtime commands in this environment.

We are planning to benchmark it according to PBBS. Therefore the input needs to be in the 2dpoints file format and the 
output needs to be in triangle file format. More on file formats are given in `geometry.md`.

The workflow:

2dpoints file -> cuda program -> triangle file

Try to keep the main kernels agnostic to file format by using an appropriate data structures internally.

## Report

Guidelines on preparint the report is given in `report_guidelines.md`