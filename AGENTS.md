## Objective

The objective of this project is to write CUDA code that constructs a **Delaunay Triangulation**, given a set of points in 2-D plane.

A sample code written is thrust is available in `sample.cu`. This is however just a guideline and the final code should be written in CUDA in 
the typical CUDA programming workflow:

1. Load the inputs from host RAM to GPU memory (using CudaMemcpy etc)
2. Launch kernels with appropriate values for blocks_per_grid, threads_per_block, amount of shared memory, streams (if any)
3. Copy the output from device memory to host memory.

The code should be readable and organized.

Complete the objective before moving on to optimizations.

The work should be done in meaningful commits.

## Optimizations

The following types of optimizations are in scope for this project:

1. Splitting work into streams which are triggered using events.
2. Using shared memory to improve performance. (use this in appropriate kernels)
3. Minimizing bank conflicts in shared memory. (only a stretch goal)

