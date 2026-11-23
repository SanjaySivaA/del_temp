#ifndef PREDICATES_CUH
#define PREDICATES_CUH

#include <cuda_runtime.h>
#include "geometry_types.h"

// -----------------------------------------------------------------------------
// Geometric Predicates & Utilities (Double Precision)
// -----------------------------------------------------------------------------

// 2D orientation test: > 0 if (a, b, c) forms a counter-clockwise (CCW) triangle,
// < 0 if clockwise (CW), and == 0 if collinear.
__device__ inline double orient2d(Point2D a, Point2D b, Point2D c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// 3x3 determinant in-circle test: returns true if point d lies strictly inside
// the circumcircle of counter-clockwise triangle (a, b, c).
__device__ inline bool is_in_circle(Point2D a, Point2D b, Point2D c, Point2D d) {
    double adx = a.x - d.x, ady = a.y - d.y;
    double bdx = b.x - d.x, bdy = b.y - d.y;
    double cdx = c.x - d.x, cdy = c.y - d.y;

    double alift = adx * adx + ady * ady;
    double blift = bdx * bdx + bdy * bdy;
    double clift = cdx * cdx + cdy * cdy;

    double det = adx * (bdy * clift - cdy * blift)
               - ady * (bdx * clift - cdx * blift)
               + alift * (bdx * cdy - cdx * bdy);

    return det > 1e-12;
}

// 32-bit Morton code (Z-order curve) interleaving from 16-bit coordinates
__device__ __host__ inline uint32_t morton2D(uint32_t x, uint32_t y) {
    x = (x | (x << 8)) & 0x00FF00FFU;
    x = (x | (x << 4)) & 0x0F0F0F0FU;
    x = (x | (x << 2)) & 0x33333333U;
    x = (x | (x << 1)) & 0x55555555U;

    y = (y | (y << 8)) & 0x00FF00FFU;
    y = (y | (y << 4)) & 0x0F0F0F0FU;
    y = (y | (y << 2)) & 0x33333333U;
    y = (y | (y << 1)) & 0x55555555U;

    return x | (y << 1);
}

// Checks if triangle tri contains edge (u, w) in any order
__device__ inline bool has_edge(int3 tri, int u, int w) {
    bool has_u = (tri.x == u || tri.y == u || tri.z == u);
    bool has_w = (tri.x == w || tri.y == w || tri.z == w);
    return has_u && has_w;
}

// Returns the vertex of triangle tri opposite to edge (u, w)
__device__ inline int get_opposite_vertex(int3 tri, int u, int w) {
    if (tri.x != u && tri.x != w) return tri.x;
    if (tri.y != u && tri.y != w) return tri.y;
    return tri.z;
}

// Returns the neighbor triangle opposite to edge (u, w)
__device__ inline int get_edge_neighbor(int3 tri, int3 nbr, int u, int w) {
    if ((tri.y == u && tri.z == w) || (tri.y == w && tri.z == u)) return nbr.x;
    if ((tri.z == u && tri.x == w) || (tri.z == w && tri.x == u)) return nbr.y;
    if ((tri.x == u && tri.y == w) || (tri.x == w && tri.y == u)) return nbr.z;
    return -1;
}

// Resolves a neighbor pointer for a triangle that was split 1-to-3
__device__ inline int resolve_split_neighbor(
    const int3* triangles,
    const int* split_base,
    int n_idx,
    int u,
    int w
) {
    if (n_idx == -1) return -1;
    int base = split_base[n_idx];
    if (base == -1) return n_idx;
    if (has_edge(triangles[n_idx], u, w)) return n_idx;
    if (has_edge(triangles[base], u, w))  return base;
    return base + 1;
}

// Resolves a neighbor pointer for a triangle that may have undergone an edge flip
__device__ inline int resolve_flipped_neighbor(
    const int3* triangles,
    const int* flip_partner,
    int n_idx,
    int u,
    int w
) {
    if (n_idx == -1) return -1;
    if (has_edge(triangles[n_idx], u, w)) return n_idx;
    int partner = flip_partner[n_idx];
    if (partner != -1 && has_edge(triangles[partner], u, w)) return partner;
    return n_idx;
}

#endif // PREDICATES_CUH
