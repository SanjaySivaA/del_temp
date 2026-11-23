#ifndef DELAUNAY_CUH
#define DELAUNAY_CUH

#include <vector>
#include <cuda_runtime.h>
#include "geometry_types.h"

// Solves 2D Delaunay Triangulation using GPU CUDA runtime workflow.
// Explicit memory management, custom kernels, shared memory optimizations,
// and streams/events synchronization.
//
// Parameters:
//   h_in_points:      Input host vector of 2D points (PBBS format or coordinates).
//   h_out_triangles:  Output host vector of Delaunay triangles (int3, 0-based indices).
//   stats:            Performance breakdown and execution statistics.
//
// Returns:
//   true on success, false on error.
bool run_delaunay_triangulation(
    const std::vector<Point2D>& h_in_points,
    std::vector<int3>& h_out_triangles,
    DelaunayStats& stats
);

#endif // DELAUNAY_CUH
