#ifndef IO_H
#define IO_H

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "geometry_types.h"

// Reads 2D points in PBBS format (or whitespace-separated coordinates)
bool read_pbbs_points(const std::string& filename, std::vector<Point2D>& points);

// Writes triangles and points in PBBS format
bool write_pbbs_triangles(
    const std::string& filename,
    const std::vector<Point2D>& points,
    const std::vector<int3>& triangles
);

// Writes simple points file (x y per line)
bool write_simple_points(const std::string& filename, const std::vector<Point2D>& points);

// Writes simple triangles file (a b c per line)
bool write_simple_triangles(const std::string& filename, const std::vector<int3>& triangles);

#endif // IO_H
