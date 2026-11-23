#ifndef GEOMETRY_TYPES_H
#define GEOMETRY_TYPES_H

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

// 2D Point structure with double precision coordinates
struct Point2D {
    double x;
    double y;

    Point2D() = default;
    __host__ __device__ Point2D(double _x, double _y) : x(_x), y(_y) {}
};

// Point with associated original index for spatial sorting and re-indexing
struct PointWithIdx {
    Point2D pt;
    int orig_idx;

    PointWithIdx() = default;
    __host__ __device__ PointWithIdx(Point2D p, int idx) : pt(p), orig_idx(idx) {}
};

// Triangle representation (3 point indices)
struct Triangle {
    int a, b, c;

    Triangle() = default;
    __host__ __device__ Triangle(int _a, int _b, int _c) : a(_a), b(_b), c(_c) {}
};

// Bounding box for 2D points
struct BoundingBox {
    Point2D min_pt;
    Point2D max_pt;

    __host__ __device__ BoundingBox() 
        : min_pt(1e30, 1e30), max_pt(-1e30, -1e30) {}
    __host__ __device__ BoundingBox(Point2D min_p, Point2D max_p)
        : min_pt(min_p), max_pt(max_p) {}
};

// Execution statistics and benchmark metrics
constexpr int MAX_FLIP_ROUNDS = 2000;

struct DelaunayStats {
    int num_points;
    int num_triangles;
    int insertion_rounds;
    int flip_rounds;
    int total_flips;
    float time_htod_ms;
    float time_sort_ms;
    float time_insertion_ms;
    float time_flip_ms;
    float time_filter_ms;
    float time_dtoh_ms;
    float time_total_gpu_ms;
};

#endif // GEOMETRY_TYPES_H
