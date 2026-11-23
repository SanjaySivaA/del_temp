#include "io.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>

bool read_pbbs_points(const std::string& filename, std::vector<Point2D>& points) {
    std::istream* in = &std::cin;
    std::ifstream file;

    if (filename != "-") {
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open input file: " << filename << std::endl;
            return false;
        }
        in = &file;
    }

    std::string token;
    if (!(*in >> token)) {
        std::cerr << "Error: Empty input file: " << filename << std::endl;
        return false;
    }

    // Check if the file starts with the PBBS header
    if (token == "pbbs_sequencePoint2d") {
        // Continue reading coordinates
    } else {
        // First token might already be the first coordinate
        try {
            double x = std::stod(token);
            double y;
            if (*in >> y) {
                points.push_back(Point2D(x, y));
            } else {
                std::cerr << "Error: Incomplete coordinate pair at file start" << std::endl;
                return false;
            }
        } catch (...) {
            std::cerr << "Warning: Unknown header token: " << token << ", attempting to parse remaining coordinates." << std::endl;
        }
    }

    double x, y;
    while (*in >> x >> y) {
        points.push_back(Point2D(x, y));
    }

    if (points.empty()) {
        std::cerr << "Error: No valid 2D points found in: " << filename << std::endl;
        return false;
    }

    return true;
}

bool write_pbbs_triangles(
    const std::string& filename,
    const std::vector<Point2D>& points,
    const std::vector<int3>& triangles
) {
    std::ostream* out = &std::cout;
    std::ofstream file;

    if (filename != "-") {
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open output file: " << filename << std::endl;
            return false;
        }
        out = &file;
    }

    // Set full double-precision output format
    out->precision(16);

    // PBBS triangles format header
    *out << "pbbs_triangles\n";
    *out << points.size() << "\n";
    *out << triangles.size() << "\n";

    // Write all points
    for (const auto& p : points) {
        *out << p.x << " " << p.y << "\n";
    }

    // Write all triangles (0-based indices)
    for (const auto& t : triangles) {
        *out << t.x << " " << t.y << " " << t.z << "\n";
    }

    return true;
}

bool write_simple_points(const std::string& filename, const std::vector<Point2D>& points) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    file.precision(16);
    for (const auto& p : points) {
        file << p.x << " " << p.y << "\n";
    }
    return true;
}

bool write_simple_triangles(const std::string& filename, const std::vector<int3>& triangles) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    for (const auto& t : triangles) {
        file << t.x << " " << t.y << " " << t.z << "\n";
    }
    return true;
}
